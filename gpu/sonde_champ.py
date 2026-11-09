"""Contacts pied-sol de MuJoCo C et de MuJoCo Warp : même modèle compilé, même état.

Champ plat (palier 2 à difficulté 0) contre plan (même config, terrain coupé),
le robot posé puis enfoncé de 3 mm. Dans l'image GPU ::

    python -m gpu.sonde_champ

Voir gpu/README.md, « Terrain du palier 2 ».
"""
from dataclasses import replace

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from gpu.poppy_env import make_env_cfg, poppy_state
from src.config import load_yaml, make_poppy_env_config

base = make_poppy_env_config(load_yaml("configs/palier2_hplaco.yaml"))
cases = {
    "champ plat": replace(base, terrain=replace(base.terrain, difficulty=0.0)),
    "plan": replace(base, terrain=replace(base.terrain, enabled=False)),
}


def describe(m, geoms, pos, dist, label, floor):
    names = [f"{m.body(m.geom_bodyid[int(g[0] if int(g[1]) in floor else g[1])]).name}#{int(g[0] if int(g[1]) in floor else g[1])}" for g in geoms]
    order = np.argsort(names)
    print(f"  {label}: {len(names)} contacts")
    for i in order:
        print(f"    {names[i]:<28} pos {np.round(pos[i], 4)}  dist {dist[i]*1000:+.2f} mm")


for label, config in cases.items():
    env = ManagerBasedRlEnv(make_env_cfg(config, num_envs=1, seed=0), device="cuda:0")
    env.reset()
    s = poppy_state(env)
    m = env.sim.mj_model
    floor = {s.floor_geom_id} | ({s.terrain_geom_id} if s.terrain is not None else set())
    for sink in (0.0, 0.003):  # posé, puis enfoncé de 3 mm
        qpos = env.sim.data.qpos[0].cpu().numpy().astype(np.float64)
        qpos[2] -= sink
        # MuJoCo C, sur le modèle compilé par mjlab.
        if s.terrain is not None:
            m.hfield_data[:] = s.terrain._grid.reshape(-1).cpu().numpy()
            m.geom_pos[s.terrain_geom_id] = env.sim.model.geom_pos[0, s.terrain_geom_id].cpu().numpy()
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        sel = [i for i in range(d.ncon) if {int(d.contact[i].geom1), int(d.contact[i].geom2)} & floor]
        print(f"== {label}, enfoncé de {sink*1000:.0f} mm")
        describe(m, [d.contact[i].geom for i in sel], [d.contact[i].pos for i in sel],
                 [d.contact[i].dist for i in sel], "MuJoCo C   ", floor)
        # MuJoCo Warp, même état.
        env.sim.data.qpos[0] = torch.as_tensor(qpos, dtype=torch.float32, device="cuda:0")
        env.sim.data.qvel[0] = 0.0
        wd = env.sim.wp_data
        mjwarp.forward(env.sim.wp_model, wd)
        n = int(wd.nacon.numpy()[0])
        geom = wd.contact.geom.numpy()[:n]
        keep = np.array([bool(set(g.tolist()) & floor) for g in geom], bool) if n else np.zeros(0, bool)
        describe(m, geom[keep], wd.contact.pos.numpy()[:n][keep], wd.contact.dist.numpy()[:n][keep],
                 "MuJoCo Warp", floor)
    env.close()
