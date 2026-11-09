"""Le portage GPU calcule-t-il la même MDP que le CPU ?

À lancer dans l'image GPU (``gpu/Dockerfile``) ::

    python -m pytest gpu/tests -q

Ces tests séparent ce qui dépend du code de ce paquet de ce qui dépend du
moteur. La récompense et l'observation sont vérifiées **sur des données
MuJoCo C** : elles doivent rendre l'``info`` CPU à l'arrondi float32 près.
La dynamique, elle, ne peut pas être identique : MuJoCo Warp génère ses
contacts plan-maillage autrement que MuJoCo C (jusqu'à 4 points par paire,
contre 3 ici). Le modèle compilé, lui, doit être identique.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("mjlab")

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from gpu.parite import STEP_TERMS, compare_models  # noqa: E402
from gpu.poppy_env import (  # noqa: E402
    compute_reward,
    make_env_cfg,
    poppy_state,
    reference_env,
)
from gpu.symmetry import mirror_action, mirror_obs, mirror_tables  # noqa: E402
from gpu.vecnormalize import RunningMeanStd  # noqa: E402
from src.config import load_yaml, make_poppy_env_config  # noqa: E402
from src.config.loaders import as_evaluation_config  # noqa: E402
from src.environments.mirror import build_mirror_spec  # noqa: E402

_CONFIG = Path("configs/exp_a_ancien_cadre_nouvelle_recompense.yaml")
_DEVICE = "cuda:0"

if not torch.cuda.is_available():
    pytest.skip("MuJoCo Warp : GPU CUDA requis", allow_module_level=True)


@pytest.fixture(scope="module")
def config():
    return make_poppy_env_config(load_yaml(_CONFIG))


@pytest.fixture(scope="module")
def env(config):
    env = ManagerBasedRlEnv(make_env_cfg(config, num_envs=4, seed=0), device=_DEVICE)
    env.reset()
    yield env
    env.close()


def test_compiled_model_matches_cpu(config, env) -> None:
    cpu = reference_env(config)
    try:
        assert compare_models(cpu.model, env.sim.mj_model) == []
    finally:
        cpu.close()


def test_reward_matches_cpu_on_mujoco_c_data(config, env) -> None:
    """compute_reward, nourri de MjData CPU, rend la récompense de step()."""
    cpu = reference_env(as_evaluation_config(config))
    cpu.reset(seed=3)
    s = poppy_state(env)
    s.reset(slice(None))
    rng = np.random.default_rng(3)

    def t(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=_DEVICE)[None].repeat(
            env.num_envs, *([1] * np.ndim(x)))

    feet = [cpu._r_foot_id, cpu._l_foot_id]
    s.feet_xy_prev[:] = t(cpu.data.xpos[feet, :2])
    worst = {k: 0.0 for k in (*STEP_TERMS, "reward")}
    for k in range(1, 301):
        action = rng.uniform(-0.6, 0.6, size=25)
        s.action[:] = t(np.clip(action, cpu.action_space.low, cpu.action_space.high))
        s.xy_before[:] = t(cpu.data.qpos[:2])
        _, reward, terminated, _, info = cpu.step(action)
        phase = torch.full((env.num_envs,), (k * cpu.dt / cpu._gait_period) % 1.0,
                           device=_DEVICE)
        gpu_reward = compute_reward(
            s,
            qpos=t(cpu.data.qpos), qvel=t(cpu.data.qvel),
            pelvis_rot=t(cpu.data.xmat[1].reshape(3, 3)),
            chest_rot=t(cpu.data.xmat[cpu._chest_id].reshape(3, 3)),
            feet_xy=t(cpu.data.xpos[feet, :2]),
            foot_force=t(np.linalg.norm(cpu.data.cfrc_ext[feet, 3:], axis=1)),
            phase=phase,
        )
        worst["reward"] = max(worst["reward"], abs(reward - gpu_reward[0].item()))
        for key in STEP_TERMS:
            worst[key] = max(worst[key], abs(float(info[key]) - s.info[key][0].item()))
        if terminated:
            break
    cpu.close()
    assert k > 50, "le robot doit tenir assez longtemps pour exercer les termes"
    bad = {key: v for key, v in worst.items() if v > 1e-3}
    assert not bad, bad


def test_reset_observation_layout(config, env) -> None:
    """Après reset : 65 dimensions, contacts nuls, horloge à phase 0."""
    obs, _ = env.reset()
    policy = obs["policy"]
    assert policy.shape == (env.num_envs, 65)
    assert torch.all(policy[:, 61:63] == 0.0)
    assert torch.allclose(policy[:, 63], torch.zeros(env.num_envs, device=_DEVICE))
    assert torch.allclose(policy[:, 64], torch.ones(env.num_envs, device=_DEVICE))
    s = poppy_state(env)
    assert torch.allclose(policy[:, 0:30], env.sim.data.qpos[:, 2:])
    assert s.healthy_z == (0.25, 0.70)
    assert env.max_episode_length == 1000


def test_actions_are_clipped_like_sb3(config, env) -> None:
    """Action bornée à l'espace annoncé (±3,1 à ±7,3), pas à [-1, 1]."""
    s = poppy_state(env)
    env.reset()
    big = torch.full((env.num_envs, 25), 10.0, device=_DEVICE)
    env.step(big)
    assert torch.allclose(s.action[0], s.action_high)
    assert float(s.action_high.min()) > 1.0  # le défaut §8 est reproduit


def test_mirror_tables_match_mirror_ppo(config, env) -> None:
    from gpu.vecnormalize import VecNormalizeWrapper

    wrapper = VecNormalizeWrapper.__new__(VecNormalizeWrapper)
    wrapper.env = env
    wrapper.device = torch.device(_DEVICE)
    wrapper.norm_obs = False
    tables = mirror_tables(wrapper)
    cpu = reference_env(config)
    spec = build_mirror_spec(cpu)
    cpu.close()
    assert np.array_equal(tables.obs_perm.cpu().numpy(), spec.obs_perm)
    assert np.array_equal(tables.joint_perm.cpu().numpy(), spec.joint_perm)
    obs = torch.randn(16, 65, device=_DEVICE)
    assert torch.allclose(mirror_obs(wrapper, mirror_obs(wrapper, obs)), obs)
    action = torch.empty(16, 25, device=_DEVICE).uniform_(-1, 1)
    expected = spec.mirror_action(action.cpu().numpy())
    assert np.allclose(mirror_action(wrapper, action).cpu().numpy(), expected, atol=1e-5)


def test_running_mean_std_matches_sb3() -> None:
    from stable_baselines3.common.running_mean_std import RunningMeanStd as SB3Rms

    ours, theirs = RunningMeanStd((7,), "cpu"), SB3Rms(shape=(7,))
    rng = np.random.default_rng(0)
    for _ in range(20):
        batch = rng.normal(3.0, 2.0, size=(int(rng.integers(1, 50)), 7))
        ours.update(torch.as_tensor(batch))
        theirs.update(batch)
    assert np.allclose(ours.mean.numpy(), theirs.mean)
    assert np.allclose(ours.var.numpy(), theirs.var)
    assert ours.count == pytest.approx(theirs.count)
    pickle.dumps(ours.state_dict())


def test_cuda_graph_update_equals_eager_update(env) -> None:
    """Le pas de mini-lot rejoué en graphe CUDA = le même pas exécuté normalement."""
    import copy

    from gpu.train import PoppyRunner, load_gpu_config, runner_cfg
    from gpu.vecnormalize import VecNormalizeWrapper

    gpu_cfg, base_cfg = load_gpu_config(Path("gpu/configs/exp_a.yaml"))
    base_cfg = copy.deepcopy(base_cfg)
    base_cfg["ppo"]["batch_size"] = 8  # 4 env × 8 pas = 32 transitions, 4 mini-lots
    wrapper = VecNormalizeWrapper(env, norm_reward=True)
    cfg = runner_cfg(base_cfg, gpu_cfg, env.num_envs, 8, 1, seed=0)
    runner = PoppyRunner(wrapper, cfg, log_dir=None, device=_DEVICE)
    runner.learn(num_learning_iterations=1)  # capture le graphe et fait une mise à jour
    alg = runner.alg
    assert alg._graphs is not None

    def snapshot() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        params = [p.detach().clone() for p in alg._params]
        state = [t.clone() for s in alg.optimizer.state.values() for t in s.values()]
        return params, state

    def restore(params: list[torch.Tensor], state: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for p, v in zip(alg._params, params, strict=True):
                p.copy_(v)
            tensors = [t for s in alg.optimizer.state.values() for t in s.values()]
            for t, v in zip(tensors, state, strict=True):
                t.copy_(v)

    alg._buffers["obs"].normal_()
    alg._buffers["actions"].normal_(0.0, 0.2)
    alg._buffers["advantages"].normal_()
    alg._buffers["returns"].normal_()
    alg._buffers["old_log_prob"].normal_(30.0, 1.0)
    start = snapshot()
    for idx in ([0, 3, 5, 7, 9, 11, 13, 15], [1, 2, 4, 6, 8, 10, 12, 14]):
        alg._idx.copy_(torch.tensor(idx, device=_DEVICE))
        alg._graphs["grad"].replay()
        alg._graphs["step"].replay()
    graph_params = [p.detach().clone() for p in alg._params]
    restore(*start)
    for idx in ([0, 3, 5, 7, 9, 11, 13, 15], [1, 2, 4, 6, 8, 10, 12, 14]):
        alg._idx.copy_(torch.tensor(idx, device=_DEVICE))
        alg._minibatch_step()
    for g, e, s in zip(graph_params, alg._params, start[0], strict=True):
        assert not torch.equal(g, s)  # la mise à jour a bien bougé les poids
        assert torch.allclose(g, e, atol=1e-7)

    # Log-vraisemblance du graphe = celle de torch.distributions, utilisée par rsl_rl.
    obs = torch.randn(16, 65, device=_DEVICE)
    act = torch.randn(16, 25, device=_DEVICE)
    mean = alg.actor.mlp(obs)
    log_std = alg._log_std()
    ours = (-0.5 * ((act - mean) * torch.exp(-log_std)) ** 2 - log_std
            - 0.5 * np.log(2 * np.pi)).sum(-1)
    ref = torch.distributions.Normal(mean, log_std.exp()).log_prob(act).sum(-1)
    assert torch.allclose(ours, ref, atol=1e-4)


def test_sb3_export_reproduces_policy(env, tmp_path) -> None:
    """Le .zip SB3 exporté rend la même action que l'acteur rsl_rl, normalisation comprise."""
    import copy

    from stable_baselines3 import PPO as SB3PPOModel
    from stable_baselines3.common.vec_env import VecNormalize

    from gpu.export_sb3 import export
    from gpu.train import PoppyRunner, load_gpu_config, runner_cfg
    from gpu.vecnormalize import VecNormalizeWrapper

    gpu_cfg, base_cfg = load_gpu_config(Path("gpu/configs/exp_a.yaml"))
    base_cfg = copy.deepcopy(base_cfg)
    base_cfg["ppo"]["batch_size"] = 8
    wrapper = VecNormalizeWrapper(env, norm_reward=True)
    runner = PoppyRunner(wrapper, runner_cfg(base_cfg, gpu_cfg, env.num_envs, 8, 1, 0),
                         log_dir=None, device=_DEVICE)
    runner.learn(num_learning_iterations=1)
    checkpoint = tmp_path / "model_0.pt"
    runner.save(str(checkpoint))
    zip_path, pkl_path = export(checkpoint, _CONFIG)

    raw = wrapper._raw_obs[:4].cpu().numpy().astype(np.float64)
    model = SB3PPOModel.load(str(zip_path), device="cpu")
    with open(pkl_path, "rb") as f:
        vec = pickle.load(f)
    assert isinstance(vec, VecNormalize)
    sb3_action, _ = model.predict(vec.normalize_obs(raw), deterministic=True)
    policy = runner.alg.get_policy()
    with torch.no_grad():
        ours = policy.mlp(wrapper.normalize_obs(wrapper._raw_obs[:4])).cpu().numpy()
    ours = np.clip(ours, model.action_space.low, model.action_space.high)
    assert np.allclose(sb3_action, ours, atol=1e-4)
