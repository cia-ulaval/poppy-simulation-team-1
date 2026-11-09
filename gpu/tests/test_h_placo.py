"""H-Placo sur GPU : même scène, même référence, même récompense, même PPO que sur CPU.

À lancer dans l'image GPU (``gpu/Dockerfile``) ::

    python -m pytest gpu/tests -q

Comme ``test_portage.py``, la récompense est vérifiée **sur des données
MuJoCo C**, ici avec l'imitation de la marche Placo et un départ au milieu
de la marche de référence. La partie PPO vérifie ce que SB3 et
``MirrorPPO`` font en plus du PPO de A : l'arrêt anticipé ``target_kl`` et
la passe de symétrie.
"""

from __future__ import annotations

import copy
from dataclasses import replace
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
    reference_start,
)
from gpu.symmetry import mirror_action, mirror_obs  # noqa: E402
from src.config import load_yaml, make_poppy_env_config  # noqa: E402
from src.config.loaders import as_evaluation_config  # noqa: E402

_CONFIG = Path("gpu/configs/h_placo_cpu.yaml")
_GPU_CONFIG = Path("gpu/configs/h_placo.yaml")
_DEVICE = "cuda:0"

if not torch.cuda.is_available():
    pytest.skip("MuJoCo Warp : GPU CUDA requis", allow_module_level=True)


@pytest.fixture(scope="module")
def config():
    return make_poppy_env_config(load_yaml(_CONFIG))


@pytest.fixture(scope="module")
def env(config):
    env = ManagerBasedRlEnv(make_env_cfg(config, num_envs=512, seed=0), device=_DEVICE)
    env.reset()
    yield env
    env.close()


def _t(x: np.ndarray, n: int) -> torch.Tensor:
    """Une valeur CPU recopiée sur ``n`` lignes, en float32 sur le GPU."""
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=_DEVICE)[None].repeat(
        n, *([1] * np.ndim(x)))


# ── Scène ──────────────────────────────────────────────────────────────────


def test_scene_with_corrected_arms_matches_cpu(config, env) -> None:
    """Bras orientés comme l'URDF, genou corrigé : modèle compilé et pose debout identiques."""
    cpu = reference_env(config)
    try:
        assert cpu._fix_arm_frames and cpu._fix_left_knee_axis
        assert compare_models(cpu.model, env.sim.mj_model) == []
        s = poppy_state(env)
        assert np.allclose(s.init_qpos.cpu().numpy(), cpu.init_qpos, atol=1e-7)
        assert np.allclose(s.joint_init.cpu().numpy(), cpu._joint_init, atol=1e-7)
    finally:
        cpu.close()


def test_arm_frames_change_the_scene(config) -> None:
    """Témoin : sans fix_arm_frames, la scène compilée n'est plus celle du CPU."""
    cpu = reference_env(config)
    other = ManagerBasedRlEnv(
        make_env_cfg(replace(config, fix_arm_frames=False, reward=replace(
            config.reward, imitation_pos_weight=0.0, imitation_upper_weight=0.0,
            imitation_vel_weight=0.0, imitation_contact_weight=0.0, imitation_base_weight=0.0),
            reference_init=0.0), num_envs=2, seed=0),
        device=_DEVICE)
    try:
        assert any(p.startswith("body_quat") for p in compare_models(cpu.model, other.sim.mj_model))
    finally:
        other.close()
        cpu.close()


def test_actions_are_bounded(config, env) -> None:
    """normalize_action_space : action bornée à [−1, 1], comme l'espace annoncé à SB3."""
    s = poppy_state(env)
    assert torch.all(s.action_low == -1.0) and torch.all(s.action_high == 1.0)
    env.reset()
    env.step(torch.full((env.num_envs, 25), 10.0, device=_DEVICE))
    assert torch.all(s.action == 1.0)


# ── Référence ──────────────────────────────────────────────────────────────


def test_reference_matches_cpu(config, env) -> None:
    """Angles, vitesses, bassin et contacts de référence = ceux de ReferenceMotion."""
    cpu = reference_env(config)
    ref_cpu = cpu._reference
    cpu.close()
    s = poppy_state(env)
    assert s.reference is not None
    assert np.allclose(s.reference.qpos.cpu().numpy(), ref_cpu.qpos, atol=1e-6)
    phases = np.concatenate([np.random.default_rng(0).uniform(0, 1, 200),
                             np.arange(ref_cpu.n_samples) / ref_cpu.n_samples])
    t = torch.as_tensor(phases, dtype=torch.float32, device=_DEVICE)
    q, qd = s.reference.joints_at(t)
    height, quat, up = s.reference.base_at(t)
    contacts = s.reference.contacts_at(t)
    for i, phase in enumerate(phases):
        q_ref, qd_ref = ref_cpu.joints_at(phase)
        h_ref, quat_ref, up_ref = ref_cpu.base_at(phase)
        assert np.allclose(q[i].cpu().numpy(), q_ref, atol=1e-5)
        assert np.allclose(qd[i].cpu().numpy(), qd_ref, atol=1e-4)
        assert abs(height[i].item() - h_ref) < 1e-6
        assert np.allclose(quat[i].cpu().numpy(), quat_ref, atol=1e-6)
        assert np.allclose(up[i].cpu().numpy(), up_ref, atol=1e-6)
        assert np.array_equal(contacts[i].cpu().numpy(), ref_cpu.contacts_at(phase))
    assert np.allclose(s.leg_mask.cpu().numpy(), cpu._leg_mask)


def test_reference_start_matches_cpu(config, env) -> None:
    """Le départ sur la référence pose le même état que _reference_start, lacet compris."""
    cpu = reference_env(config)
    cpu.reset(seed=5)
    s = poppy_state(env)
    rng = np.random.default_rng(5)
    for _ in range(20):
        qpos = cpu.init_qpos + rng.uniform(-0.01, 0.01, cpu.model.nq)
        qvel = cpu.init_qvel + rng.uniform(-0.01, 0.01, cpu.model.nv)
        half = 0.5 * rng.uniform(-0.26, 0.26)
        qpos[3:7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
        q_gpu, v_gpu = _t(qpos, 1), _t(qvel, 1)
        phase = cpu._reference_start(qpos, qvel)
        reference_start(s, q_gpu, v_gpu, torch.tensor([phase], device=_DEVICE))
        assert np.allclose(q_gpu[0].cpu().numpy(), qpos, atol=2e-6)
        assert np.allclose(v_gpu[0].cpu().numpy(), qvel, atol=2e-5)
    cpu.close()


def test_reset_starts_on_reference_with_its_phase(config, env) -> None:
    """reference_init = 0,75 : trois départs sur quatre sur la marche, horloge et filtre suivent."""
    obs, _ = env.reset()
    s = poppy_state(env)
    started = s.phase_offset > 0.0
    share = started.float().mean().item()
    assert 0.68 < share < 0.82
    policy = obs["policy"]
    angle = 2.0 * np.pi * s.phase_offset
    assert torch.allclose(policy[:, 63], torch.sin(angle), atol=1e-5)
    assert torch.allclose(policy[:, 64], torch.cos(angle), atol=1e-5)
    speed = s.reference.speed
    assert torch.allclose(s.forward_vel_filtered[started], torch.full_like(
        s.forward_vel_filtered[started], speed))
    assert torch.all(s.forward_vel_filtered[~started] == 0.0)
    # Hauteur du bassin : celle de la référence à la phase tirée, pour les départs sur la marche.
    height, _, _ = s.reference.base_at(s.phase_offset)
    assert torch.allclose(env.sim.data.qpos[started, 2], height[started], atol=1e-5)
    standing = s.init_qpos[2]
    assert torch.all((env.sim.data.qpos[~started, 2] - standing).abs() <= 0.0100001)


# ── Récompense ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("start_on_reference", [False, True])
def test_reward_matches_cpu_on_mujoco_c_data(config, env, start_on_reference) -> None:
    """compute_reward, nourri de MjData CPU, rend la récompense H-Placo terme pour terme.

    Imitation comprise ; avec ``start_on_reference``, l'épisode part au
    milieu de la marche de référence (phase et vitesse filtrée de départ).
    """
    cpu_config = as_evaluation_config(config)
    if start_on_reference:
        cpu_config = replace(cpu_config, reference_init=1.0)
    cpu = reference_env(cpu_config)
    cpu.reset(seed=11)
    n = 4
    s = copy.copy(poppy_state(env))
    s.allocate(n, _DEVICE)
    start = cpu._phase_time / cpu._gait_period
    s.start_forward_vel[:] = cpu._forward_vel_filtered
    s.reset(slice(None))
    rng = np.random.default_rng(11)
    feet = [cpu._r_foot_id, cpu._l_foot_id]
    s.feet_xy_prev[:] = _t(cpu.data.xpos[feet, :2], n)
    terms = (*STEP_TERMS, "imitation_reward", "reward")
    worst = dict.fromkeys(terms, 0.0)
    imitation = []
    for k in range(1, 301):
        action = rng.uniform(-0.3, 0.3, size=25)
        s.action[:] = _t(np.clip(action, -1.0, 1.0), n)
        s.xy_before[:] = _t(cpu.data.qpos[:2], n)
        _, reward, terminated, _, info = cpu.step(action)
        phase = torch.full((n,), (start + k * cpu.dt / cpu._gait_period) % 1.0, device=_DEVICE)
        gpu_reward = compute_reward(
            s,
            qpos=_t(cpu.data.qpos, n), qvel=_t(cpu.data.qvel, n),
            pelvis_rot=_t(cpu.data.xmat[1].reshape(3, 3), n),
            chest_rot=_t(cpu.data.xmat[cpu._chest_id].reshape(3, 3), n),
            feet_xy=_t(cpu.data.xpos[feet, :2], n),
            foot_force=_t(np.linalg.norm(cpu.data.cfrc_ext[feet, 3:], axis=1), n),
            phase=phase,
        )
        info = {**info, "reward": reward}
        for key in terms:
            value = gpu_reward[0].item() if key == "reward" else s.info[key][0].item()
            worst[key] = max(worst[key], abs(float(info[key]) - value))
        imitation.append(float(info["imitation_reward"]))
        if terminated:
            break
    cpu.close()
    assert k > 30, "le robot doit tenir assez longtemps pour exercer les termes"
    assert np.mean(imitation) > 0.5, "l'imitation doit peser dans la récompense"
    bad = {key: v for key, v in worst.items() if v > 1e-3}
    assert not bad, bad


# ── PPO : target_kl et passe de symétrie ───────────────────────────────────


@pytest.fixture(scope="module")
def runner(env):
    from gpu.train import PoppyRunner, load_gpu_config, runner_cfg
    from gpu.vecnormalize import VecNormalizeWrapper

    gpu_cfg, base_cfg = load_gpu_config(_GPU_CONFIG)
    base_cfg = copy.deepcopy(base_cfg)
    base_cfg["ppo"]["batch_size"] = 1024  # 512 env × 8 pas = 4096 transitions, 4 mini-lots
    wrapper = VecNormalizeWrapper(env, norm_reward=True)
    cfg = runner_cfg(base_cfg, gpu_cfg, env.num_envs, 8, 1, seed=0)
    assert cfg["algorithm"]["target_kl"] == 0.03
    assert cfg["algorithm"]["sb3_symmetry_weight"] == 1.0
    runner = PoppyRunner(wrapper, cfg, log_dir=None, device=_DEVICE)
    runner.learn(num_learning_iterations=1)  # capture les graphes, une mise à jour
    return runner


def test_symmetry_pass_mirrors_like_mirror_ppo(runner) -> None:
    """Les reflets du pas de symétrie (tampons statiques) = ceux de gpu.symmetry."""
    alg, wrapper = runner.alg, runner.env
    assert set(alg._graphs) == {"grad", "step", "sym"}
    b = alg._buffers
    b["obs_mean"].copy_(wrapper.obs_rms.mean.to(torch.float32))
    b["obs_std"].copy_(torch.sqrt(wrapper.obs_rms.var + wrapper.epsilon).to(torch.float32))
    obs = torch.randn(64, 65, device=_DEVICE)
    assert torch.allclose(alg._mirror_obs(obs), mirror_obs(wrapper, obs), atol=1e-5)
    action = torch.empty(64, 25, device=_DEVICE).uniform_(-1.2, 1.2)
    assert torch.allclose(alg._mirror_action(action), mirror_action(wrapper, action), atol=1e-6)


def test_symmetry_step_touches_only_the_mean_network(runner) -> None:
    """Comme dans SB3 : log_std et critique sans gradient, leur Adam n'avance pas."""
    alg = runner.alg
    rest_before = [p.detach().clone() for p in alg._rest_params]
    rest_state = [t.clone() for st in alg.optimizer.parts[1].state.values() for t in st.values()]
    mean_before = [p.detach().clone() for p in alg._mean_params]
    alg._buffers["obs"].normal_()
    alg._idx.copy_(torch.arange(alg._idx.numel(), device=_DEVICE))
    alg._graphs["sym"].replay()
    assert all(torch.equal(a, p) for a, p in zip(rest_before, alg._rest_params, strict=True))
    after = [t for st in alg.optimizer.parts[1].state.values() for t in st.values()]
    assert all(torch.equal(a, b) for a, b in zip(rest_state, after, strict=True))
    assert any(not torch.equal(a, p) for a, p in zip(mean_before, alg._mean_params, strict=True))


def test_symmetry_graph_equals_eager_step(runner) -> None:
    """Le pas de symétrie rejoué en graphe = le même pas exécuté normalement."""
    alg = runner.alg
    params = [p.detach().clone() for p in alg._params]
    state = [t.clone() for st in alg.optimizer.state.values() for t in st.values()]

    def restore() -> None:
        with torch.no_grad():
            for p, v in zip(alg._params, params, strict=True):
                p.copy_(v)
            tensors = [t for st in alg.optimizer.state.values() for t in st.values()]
            for t, v in zip(tensors, state, strict=True):
                t.copy_(v)

    alg._buffers["obs"].normal_()
    alg._idx.copy_(torch.arange(alg._idx.numel(), device=_DEVICE))
    alg._graphs["sym"].replay()
    graph = [p.detach().clone() for p in alg._params]
    restore()
    alg._sym_step()
    for g, e in zip(graph, alg._params, strict=True):
        assert torch.allclose(g, e, atol=1e-7)
    restore()


def test_target_kl_stops_the_update(runner) -> None:
    """Divergence au-delà de 1,5 × target_kl : le pas n'est pas fait, la mise à jour s'arrête."""
    alg = runner.alg
    b = alg._buffers
    alg._idx.copy_(torch.arange(alg._idx.numel(), device=_DEVICE))
    alg._graphs["grad"].replay()
    log_std = alg._log_std()
    obs = b["obs"][alg._idx]
    with torch.no_grad():
        mean = alg.actor.mlp(obs)
        act = b["actions"][alg._idx]
        log_prob = torch.distributions.Normal(mean, log_std.exp()).log_prob(act).sum(-1)
        log_ratio = log_prob - b["old_log_prob"][alg._idx]
        expected = ((log_ratio.exp() - 1.0) - log_ratio).mean()
    assert torch.allclose(b["kl"], expected, rtol=1e-4, atol=1e-7)

    # Borne minuscule : la première divergence (arrondis float32) la dépasse,
    # aucune époque n'est complète.
    stats = []
    update, saved = alg.update, alg.target_kl

    def recording_update() -> dict[str, float]:
        out = update()
        stats.append(out)
        return out

    try:
        alg.target_kl = 1e-12
        alg.update = recording_update
        runner.learn(num_learning_iterations=1)
    finally:
        alg.update, alg.target_kl = update, saved
    assert stats and stats[0]["epochs_done"] == 0.0

    # Borne de H-Placo (0,03) : une mise à jour normale va au bout ou s'arrête proprement.
    stats.clear()
    try:
        alg.update = recording_update
        runner.learn(num_learning_iterations=1)
    finally:
        alg.update = update
    assert 0.0 <= stats[0]["epochs_done"] <= alg.num_learning_epochs
    assert stats[0]["symmetry"] >= 0.0
