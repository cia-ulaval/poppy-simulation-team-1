import mujoco
import numpy as np


def cast_horizontal_rays(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_rays: int = 64,
    fov: float = np.pi / 2,
    max_distance: float = 10.0,
    head_offset: np.ndarray = None,
) -> np.ndarray:
    if head_offset is None:
        head_offset = np.array([0.0, 0.0, 0.19])

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    if torso_id == -1:
        raise ValueError("Could not find 'torso' body in model")

    torso_pos = data.xpos[torso_id].copy()
    torso_mat = data.xmat[torso_id].reshape(3, 3).copy()

    head_pos = torso_pos + torso_mat @ head_offset

    vec = np.zeros((n_rays, 3), dtype=np.float64)

    angles = np.linspace(-fov / 2, fov / 2, n_rays)
    for i, angle in enumerate(angles):
        local_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
        world_dir = torso_mat @ local_dir
        vec[i] = world_dir / np.linalg.norm(world_dir)

    geomgroup = None
    flg_static = 1
    bodyexclude = torso_id

    geomid = np.zeros(n_rays, dtype=np.int32)
    distances = np.zeros(n_rays, dtype=np.float64)
    normal = np.zeros((n_rays, 3), dtype=np.float64)

    mujoco.mj_multiRay(
        model,
        data,
        head_pos,
        vec.flatten(),
        geomgroup,
        flg_static,
        bodyexclude,
        geomid,
        distances,
        normal.flatten(),
        n_rays,
        max_distance,
    )

    distances[distances < 0] = max_distance

    return distances.reshape(8, 8).min(axis=1)
