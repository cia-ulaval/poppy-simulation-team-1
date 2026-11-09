"""Palier 2 sur GPU : même relief, même sol, même récompense sur terrain que sur CPU.

À lancer dans l'image GPU (``gpu/Dockerfile``) ::

    python -m pytest gpu/tests -q

Le CPU tire son relief avec le générateur numpy de l'environnement, le GPU
avec celui de torch : les tirages ne peuvent pas coïncider. Ce qui est
vérifié, c'est le reste. Pour des tirages donnés, le relief est le même case
par case. La surface que touche le robot est la même, mesurée par MuJoCo C
sur la scène en tuiles. La hauteur du sol et la récompense sont les mêmes sur
données MuJoCo C.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

pytest.importorskip("mjlab")

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from gpu.parite import STEP_TERMS, compare_models  # noqa: E402
from gpu.poppy_env import compute_reward, make_env_cfg, poppy_state, reference_env  # noqa: E402
from gpu.terrain import TerrainDraws, build_heights, cells_per_side  # noqa: E402
from src.config import load_yaml, make_poppy_env_config  # noqa: E402
from src.config.loaders import as_evaluation_config  # noqa: E402
from src.environments.terrain import TERRAIN_NAME, generate_heights  # noqa: E402

_CONFIG = Path("configs/palier2_hplaco.yaml")
_GPU_CONFIG = Path("gpu/configs/palier2_hplaco.yaml")
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


def _numpy_draws(rng: np.random.Generator, n: int, half: float) -> TerrainDraws:
    """Les tirages de ``generate_heights``, dans son ordre exact, en float64."""
    waves = np.array([[rng.uniform(3.0, 8.0), rng.uniform(-np.pi, np.pi),
                       rng.uniform(-np.pi, np.pi)] for _ in range(2)])
    bumps = []
    for _ in range(80):
        cx, cy = rng.uniform(-half, half, size=2)
        bumps.append([cx, cy, rng.uniform(0.15, 0.5), rng.uniform(-1.0, 1.0)])
    steps = []
    for _ in range(60):
        cx, cy = rng.uniform(-half, half, size=2)
        wx, wy = rng.uniform(0.3, 1.2, size=2)
        steps.append([cx, cy, wx, wy, rng.uniform(-1.0, 1.0)])
    rough = rng.uniform(-1.0, 1.0, size=(n, n))
    bumps, steps = np.array(bumps), np.array(steps)

    def t(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float64, device=_DEVICE)[None]

    return TerrainDraws(
        wavelength=t(waves[:, 0]), direction=t(waves[:, 1]), phase=t(waves[:, 2]),
        bump_xy=t(bumps[:, 0:2]), bump_sigma=t(bumps[:, 2]), bump_amp=t(bumps[:, 3]),
        step_xy=t(steps[:, 0:2]), step_wh=t(steps[:, 2:4]), step_amp=t(steps[:, 4]),
        rough=t(rough),
    )


# ── Relief ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("difficulty", [0.1, 0.6, 1.0])
def test_same_draws_give_the_cpu_relief(config, difficulty) -> None:
    """Mêmes tirages, même difficulté : le relief de generate_heights, case par case."""
    n = cells_per_side(config.terrain)
    for seed in (0, 1, 2):
        expected = generate_heights(config.terrain, difficulty, n, np.random.default_rng(seed))
        draws = _numpy_draws(np.random.default_rng(seed), n, 0.5 * config.terrain.size)
        ours = build_heights(draws, config.terrain, difficulty, n)[0].cpu().numpy()
        assert np.abs(ours - expected).max() < 1e-9
        assert np.abs(expected).max() > 0.01 * difficulty  # un vrai relief


def test_flat_at_zero_difficulty(config) -> None:
    n = cells_per_side(config.terrain)
    draws = _numpy_draws(np.random.default_rng(0), n, 10.0)
    assert torch.all(build_heights(draws, config.terrain, 0.0, n) == 0.0)


def test_relief_statistics_match_cpu(config) -> None:
    """Tirages torch contre tirages numpy : même loi (écart type, extrêmes, part plate)."""
    n = cells_per_side(config.terrain)
    cpu = np.stack([generate_heights(config.terrain, 1.0, n, np.random.default_rng(s))
                    for s in range(24)])
    torch.manual_seed(0)
    draws = TerrainDraws.sample(24, n, 0.5 * config.terrain.size, _DEVICE)
    gpu = build_heights(draws, config.terrain, 1.0, n).cpu().numpy()
    assert abs(gpu.std() - cpu.std()) < 0.1 * cpu.std()
    assert abs(np.abs(gpu).max() - np.abs(cpu).max()) < 0.3 * np.abs(cpu).max()
    assert abs(np.mean(gpu == 0.0) - np.mean(cpu == 0.0)) < 0.01


# ── Scène et surface ───────────────────────────────────────────────────────


def test_scene_matches_cpu_except_the_field_size(config, env) -> None:
    """La scène en tuiles est celle du CPU ; seuls la taille du champ et son geom diffèrent."""
    cpu = reference_env(config)
    try:
        problems = compare_models(cpu.model, env.sim.mj_model)
        terrain = cpu.model.geom(TERRAIN_NAME).id
        assert all(p.startswith(("geom_size", "geom_aabb", "geom_rbound")) for p in problems), problems
        mask = np.ones(cpu.model.ngeom, bool)
        mask[terrain] = False
        assert np.allclose(cpu.model.geom_size[mask], env.sim.mj_model.geom_size[mask])
        for field in ("geom_friction", "geom_solref", "geom_solimp", "geom_condim",
                      "geom_contype", "geom_conaffinity", "geom_pos", "geom_quat"):
            assert np.allclose(getattr(cpu.model, field)[terrain],
                               getattr(env.sim.mj_model, field)[terrain]), field
        for field in ("hfield_size",):
            a, b = getattr(cpu.model, field)[0], getattr(env.sim.mj_model, field)[0]
            assert np.allclose(a[2:], b[2:])  # même hauteur et même socle
    finally:
        cpu.close()


def test_tile_surface_is_the_cpu_surface(config, env) -> None:
    """Un relief CPU écrit dans une tuile : même surface sous le robot, mesurée par MuJoCo C.

    La scène en tuiles compilée par mjlab est simulée ici en MuJoCo C, avec
    les données du champ et la position du geom du monde voulu : cela vérifie
    l'agencement (lignes = y, colonnes = x, décalage du geom), indépendamment
    de MuJoCo Warp.
    """
    cpu = reference_env(replace(config, terrain=replace(config.terrain, difficulty=1.0)))
    cpu.reset(seed=4)
    heights = cpu._terrain.heights.copy()
    s = poppy_state(env)
    world = 2
    ids = torch.tensor([world], device=_DEVICE)
    s.terrain.write(ids, torch.as_tensor(heights, dtype=torch.float32, device=_DEVICE)[None])

    m = copy.copy(env.sim.mj_model)
    terrain = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "robot/terrain")
    m.hfield_data[:] = s.terrain._grid.reshape(-1).cpu().numpy()
    m.geom_pos[terrain] = env.sim.model.geom_pos[world, terrain].cpu().numpy()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cpu_terrain = cpu.model.geom(TERRAIN_NAME).id

    rng = np.random.default_rng(0)
    down = np.array([0.0, 0.0, -1.0])
    worst_surface, worst_lookup = 0.0, 0.0
    xy = rng.uniform(-9.5, 9.5, size=(300, 2))
    for x, y in xy:
        start = np.array([x, y, 1.0])
        z_cpu = 1.0 - mujoco.mj_rayHfield(cpu.model, cpu.data, cpu_terrain, start, down)
        z_gpu = 1.0 - mujoco.mj_rayHfield(m, d, terrain, start, down)
        worst_surface = max(worst_surface, abs(z_cpu - z_gpu))
    lookup = s.terrain.height_at(ids.repeat(len(xy)), torch.as_tensor(xy, dtype=torch.float32,
                                                                      device=_DEVICE))
    for (x, y), h in zip(xy, lookup.cpu().numpy(), strict=True):
        worst_lookup = max(worst_lookup, abs(cpu._terrain.height_at(x, y) - h))
    cpu.close()
    assert np.abs(heights).max() > 0.02
    assert worst_surface < 1e-5, worst_surface
    assert worst_lookup < 1e-5, worst_lookup


def test_reset_draws_one_tile_per_env_with_a_flat_start(config, env) -> None:
    """Au reset, chaque environnement a son relief, plat dans la zone de départ."""
    s = poppy_state(env)
    s.terrain.difficulty = 1.0
    try:
        env.reset()
        ids = torch.arange(env.num_envs, device=_DEVICE)
        tiles = s.terrain.tiles(ids)
        assert not torch.allclose(tiles[0], tiles[1])
        n = s.terrain.layout.n
        axis = torch.linspace(-10.0, 10.0, n, device=_DEVICE)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        start = torch.hypot(xx, yy) < config.terrain.flat_radius
        assert torch.allclose(tiles[:, start], torch.full_like(tiles[:, start], 0.5))
        offsets = env.sim.model.geom_pos[:, s.terrain_geom_id, 0:2]
        assert torch.allclose(offsets, s.terrain_offset)
        assert len({tuple(o) for o in offsets.cpu().numpy().round(3).tolist()}) == env.num_envs
    finally:
        s.terrain.difficulty = config.terrain.difficulty


def test_contacts_land_on_the_world_tile(env) -> None:
    """Les contacts de MuJoCo Warp sont pris sur la tuile du monde, pas ailleurs dans le champ.

    MuJoCo Warp ne calcule geom_xpos des geoms statiques qu'une fois, dans
    make_data : déplacer le champ par geom_pos ne suffit pas. Chaque monde
    reçoit ici une tuile plate à sa propre hauteur, 0, -1, -2 et -3 cm, et ses
    contacts doivent s'y trouver.
    """
    s = poppy_state(env)
    env.reset()
    ids = torch.arange(env.num_envs, device=_DEVICE)
    assert torch.allclose(env.sim.data.geom_xpos[:, s.terrain_geom_id, 0:2], s.terrain_offset)
    n = s.terrain.layout.n
    level = -0.01 * ids.to(torch.float32)
    s.terrain.write(ids, level[:, None, None].expand(-1, n, n).contiguous())
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=_DEVICE)
    wd = env.sim.wp_data
    seen = {i: [] for i in range(env.num_envs)}
    for _ in range(30):
        env.step(zero)
        count = int(wd.nacon.numpy()[0])
        geom = wd.contact.geom.numpy()[:count]
        on_field = (geom == s.terrain_geom_id).any(axis=1)
        z = wd.contact.pos.numpy()[:count, 2][on_field]
        for world, height in zip(wd.contact.worldid.numpy()[:count][on_field], z, strict=True):
            seen[int(world)].append(float(height))
    for world, heights in seen.items():
        assert len(heights) > 10, world
        assert abs(np.median(heights) - level[world].item()) < 0.003, (world, np.median(heights))


# ── Récompense sur terrain ─────────────────────────────────────────────────


def test_reward_matches_cpu_on_terrain(config, env) -> None:
    """compute_reward, nourri de MjData CPU sur relief, rend la récompense terme pour terme.

    Sans zone plate et à la difficulté 1 : le robot part au-dessus d'un sol qui
    n'est pas à zéro, ce qui exerce ``healthy`` et l'imitation de la hauteur.
    """
    terrain = replace(config.terrain, difficulty=1.0, flat_radius=0.0)
    cpu_config = replace(as_evaluation_config(replace(config, terrain=terrain),
                                              terrain_difficulty=1.0), reference_init=1.0)
    cpu = reference_env(cpu_config)
    for seed in range(6, 60):  # un départ au-dessus d'un relief d'au moins 1 cm
        cpu.reset(seed=seed)
        if abs(cpu._ground_height()) > 0.01:
            break
    n = 4
    s = copy.copy(poppy_state(env))
    s.allocate(n, _DEVICE)
    start = cpu._phase_time / cpu._gait_period
    s.start_forward_vel[:] = cpu._forward_vel_filtered
    s.reset(slice(None))
    rng = np.random.default_rng(6)

    def t(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=_DEVICE)[None].repeat(
            n, *([1] * np.ndim(x)))

    feet = [cpu._r_foot_id, cpu._l_foot_id]
    s.feet_xy_prev[:] = t(cpu.data.xpos[feet, :2])
    terms = (*STEP_TERMS, "imitation_reward", "reward")
    worst = dict.fromkeys(terms, 0.0)
    grounds = []
    for k in range(1, 301):
        action = rng.uniform(-0.3, 0.3, size=25)
        s.action[:] = t(action)
        s.xy_before[:] = t(cpu.data.qpos[:2])
        _, reward, terminated, _, info = cpu.step(action)
        ground = cpu._ground_height()
        grounds.append(ground)
        phase = torch.full((n,), (start + k * cpu.dt / cpu._gait_period) % 1.0, device=_DEVICE)
        gpu_reward = compute_reward(
            s, qpos=t(cpu.data.qpos), qvel=t(cpu.data.qvel),
            pelvis_rot=t(cpu.data.xmat[1].reshape(3, 3)),
            chest_rot=t(cpu.data.xmat[cpu._chest_id].reshape(3, 3)),
            feet_xy=t(cpu.data.xpos[feet, :2]),
            foot_force=t(np.linalg.norm(cpu.data.cfrc_ext[feet, 3:], axis=1)),
            phase=phase, ground=torch.full((n,), ground, device=_DEVICE),
        )
        info = {**info, "reward": reward}
        for key in terms:
            value = gpu_reward[0].item() if key == "reward" else s.info[key][0].item()
            worst[key] = max(worst[key], abs(float(info[key]) - value))
        if terminated:
            break
    cpu.close()
    assert k > 30
    assert max(abs(g) for g in grounds) > 0.005, "le sol doit être en relief sous le robot"
    bad = {key: v for key, v in worst.items() if v > 1e-3}
    assert not bad, bad


# ── Entraînement : curriculum et --init-from ───────────────────────────────


def test_curriculum_follows_cpu_schedule(env) -> None:
    """Difficulté réglée avant chaque collecte, comme CurriculumCallback."""
    from gpu.train import PoppyRunner, install_curriculum, load_gpu_config, runner_cfg
    from gpu.vecnormalize import VecNormalizeWrapper
    from src.training.curriculum import CurriculumConfig

    gpu_cfg, base_cfg = load_gpu_config(_GPU_CONFIG)
    base_cfg = copy.deepcopy(base_cfg)
    base_cfg["ppo"]["batch_size"] = 16  # 4 env × 8 pas = 32 transitions
    wrapper = VecNormalizeWrapper(env, norm_reward=True)
    runner = PoppyRunner(wrapper, runner_cfg(base_cfg, gpu_cfg, env.num_envs, 8, 2, 0),
                         log_dir=None, device=_DEVICE)
    curriculum = CurriculumConfig.from_yaml(base_cfg["curriculum"])
    total = 320
    install_curriculum(runner, curriculum, total)
    s = poppy_state(env)
    assert s.terrain.difficulty == pytest.approx(0.1)
    runner.learn(num_learning_iterations=2)
    expected = curriculum.at(64 / total)["terrain_difficulty"]
    assert s.terrain.difficulty == pytest.approx(expected)
    s.terrain.difficulty = 0.1


def test_init_from_sb3_restores_policy_adam_and_normalisation(env, tmp_path) -> None:
    """Un modèle SB3 repris : même action, même état d'Adam, mêmes statistiques d'observation."""
    from stable_baselines3 import PPO

    from gpu.export_sb3 import export
    from gpu.import_sb3 import load_into
    from gpu.train import PoppyRunner, load_gpu_config, runner_cfg
    from gpu.vecnormalize import VecNormalizeWrapper

    gpu_cfg, base_cfg = load_gpu_config(_GPU_CONFIG)
    base_cfg = copy.deepcopy(base_cfg)
    base_cfg["ppo"]["batch_size"] = 16
    cfg = runner_cfg(base_cfg, gpu_cfg, env.num_envs, 8, 1, 0)

    # Un modèle SB3 avec un état d'Adam : export d'un runner, puis un pas d'Adam.
    first = PoppyRunner(VecNormalizeWrapper(env, norm_reward=True), copy.deepcopy(cfg),
                        log_dir=None, device=_DEVICE)
    first.learn(num_learning_iterations=1)
    checkpoint = tmp_path / "model_0.pt"
    first.save(str(checkpoint))
    zip_path, pkl_path = export(checkpoint, _CONFIG)
    folder = tmp_path / "best_model"
    folder.mkdir()
    model = PPO.load(str(zip_path), device="cpu")
    obs = torch.randn(32, 65)
    loss = model.policy.get_distribution(obs).distribution.mean.pow(2).mean() \
        + model.policy.predict_values(obs).pow(2).mean() + model.policy.log_std.pow(2).sum()
    model.policy.optimizer.zero_grad()
    loss.backward()
    model.policy.optimizer.step()
    model.save(str(folder / "best_model.zip"))
    (folder / "vec_normalize.pkl").write_bytes(pkl_path.read_bytes())

    second = PoppyRunner(VecNormalizeWrapper(env, norm_reward=True), copy.deepcopy(cfg),
                         log_dir=None, device=_DEVICE)
    copied = load_into(second, folder / "best_model.zip")
    assert copied == {"params": 13, "adam_states": 13}
    raw = obs.numpy().astype(np.float64)
    sb3_action, _ = model.predict(raw, deterministic=True)
    with torch.no_grad():
        ours = second.alg.actor.mlp(obs.to(_DEVICE)).cpu().numpy()
    assert np.allclose(np.clip(ours, -1, 1), sb3_action, atol=1e-5)
    assert torch.allclose(second.env.obs_rms.mean.cpu(), first.env.obs_rms.mean.cpu())
    sb3_state = model.policy.optimizer.state_dict()["state"]
    log_std = second.alg.actor.distribution.log_std_param
    names = [n for n, _ in model.policy.named_parameters()]
    index = names.index("log_std")
    state = second.alg.optimizer.owner(log_std).state[log_std]
    assert torch.allclose(state["exp_avg"].cpu(), sb3_state[index]["exp_avg"])
    assert float(state["step"]) == float(sb3_state[index]["step"])
    # La capture du graphe garde cet état (elle ne le remet pas à zéro).
    second.learn(num_learning_iterations=1)
    assert float(second.alg.optimizer.owner(log_std).state[log_std]["step"]) > 1.0
