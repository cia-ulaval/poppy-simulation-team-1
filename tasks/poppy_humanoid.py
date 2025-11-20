"""
Poppy Humanoid Isaac Gym Environment for Walking
Optimized for massively parallel training on GPU
"""

import numpy as np
import os

# CRITICAL: Import isaacgym BEFORE torch
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *

# Now safe to import torch
import torch


class PoppyHumanoid:
    """
    Isaac Gym environment for Poppy Humanoid robot learning to walk
    """

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        """
        Initialize Poppy Humanoid environment

        Args:
            cfg: Configuration dictionary from YAML
            sim_device: Device for simulation ('cuda:0' or 'cpu')
            graphics_device_id: GPU ID for graphics
            headless: Whether to run without visualization
        """
        self.cfg = cfg

        # Auto-detect if CUDA is available
        self.use_gpu = torch.cuda.is_available() and 'cuda' in sim_device
        if not self.use_gpu:
            print("[Warning] CUDA not available or not requested, using CPU mode")
            sim_device = 'cpu'

        self.sim_device = sim_device
        self.graphics_device_id = graphics_device_id
        self.headless = headless

        # Environment parameters from config
        self.num_envs = cfg["env"]["numEnvs"]
        self.num_obs = cfg["env"]["numObservations"]
        self.num_actions = cfg["env"]["numActions"]
        self.max_episode_length = cfg["env"]["episodeLength"]

        # Reward scales from config
        self.reward_scales = cfg["env"]["learn"]["rewardScales"]

        # Physics parameters
        self.dt = cfg["sim"]["dt"]
        self.control_freq = cfg["env"]["controlFrequencyInv"]

        # Episode tracking
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.sim_device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.long, device=self.sim_device)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.sim_device)

        # Poppy-specific parameters
        self.base_init_pos = [0, 0, 0.98]  # Initial height for Poppy
        self.initial_dof_pos = torch.zeros(self.num_actions, dtype=torch.float, device=self.sim_device)

        # Create gym and sim
        self.gym = gymapi.acquire_gym()
        self.sim = self._create_sim()
        self._create_ground_plane()
        self._create_envs()

        # Prepare simulation tensors
        self.gym.prepare_sim(self.sim)

        # Get gym state tensors
        self._init_buffers()

        # Initialize observations and actions
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.sim_device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.sim_device, dtype=torch.float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.sim_device, dtype=torch.float)
        self.last_actions = torch.zeros((self.num_envs, self.num_actions), device=self.sim_device, dtype=torch.float)

        print(f"Poppy Humanoid environment initialized with {self.num_envs} environments")

    def _create_sim(self):
        """Create Isaac Gym simulation"""
        sim_params = gymapi.SimParams()

        # Physics parameters
        sim_params.dt = self.dt
        sim_params.substeps = self.cfg["sim"]["substeps"]
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        # PhysX parameters
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.max_depenetration_velocity = 1.0

        # Only use GPU pipeline if CUDA is available
        sim_params.use_gpu_pipeline = self.use_gpu

        sim = self.gym.create_sim(
            compute_device=0 if self.use_gpu else -1,
            graphics_device=self.graphics_device_id if self.use_gpu else -1,
            type=gymapi.SIM_PHYSX,
            params=sim_params
        )

        if sim is None:
            raise Exception("Failed to create sim")

        return sim

    def _create_ground_plane(self):
        """Create ground plane"""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """Create parallel environments with Poppy robot"""
        spacing = self.cfg["env"]["envSpacing"]
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Load Poppy URDF
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = False
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.density = 1000.0
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.armature = 0.01
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        # Use default shapes if meshes are not found
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        asset_options.override_com = True
        asset_options.override_inertia = True

        poppy_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Get DOF properties
        dof_props = self.gym.get_asset_dof_properties(poppy_asset)
        self.num_dof = self.gym.get_asset_dof_count(poppy_asset)

        # Set DOF properties (PD gains)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        dof_props["stiffness"].fill(0.0)
        dof_props["damping"].fill(0.0)

        # Set effort limits
        dof_props["effort"][:] = 300.0

        # Initial DOF positions (standing pose)
        self.initial_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device)

        # Create environments
        self.envs = []
        self.actor_handles = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, int(np.sqrt(self.num_envs)))

            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(*self.base_init_pos)
            start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            actor_handle = self.gym.create_actor(env, poppy_asset, start_pose, "poppy", i, 1, 0)

            self.gym.set_actor_dof_properties(env, actor_handle, dof_props)

            self.envs.append(env)
            self.actor_handles.append(actor_handle)

        print(f"Created {self.num_envs} environments with Poppy Humanoid ({self.num_dof} DOFs)")

    def _init_buffers(self):
        """Initialize state tensors from simulation"""
        # Get state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # Create views into state tensors
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # Base position, rotation, velocities
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = self.root_states[:, 7:10]
        self.base_ang_vel = self.root_states[:, 10:13]

        # Initial state backup
        self.initial_root_states = self.root_states.clone()
        self.initial_dof_pos_tensor = self.dof_pos.clone()
        self.initial_dof_vel = self.dof_vel.clone()

    def reset_idx(self, env_ids):
        """Reset specific environments"""
        num_resets = len(env_ids)

        # Reset DOF states
        self.dof_pos[env_ids] = self.initial_dof_pos_tensor[env_ids] + \
                                0.1 * (torch.rand((num_resets, self.num_dof), device=self.sim_device) - 0.5)
        self.dof_vel[env_ids] = 0.1 * (torch.rand((num_resets, self.num_dof), device=self.sim_device) - 0.5)

        # Reset root states
        self.root_states[env_ids] = self.initial_root_states[env_ids].clone()
        self.root_states[env_ids, 0:3] = torch.tensor(self.base_init_pos, device=self.sim_device)
        self.root_states[env_ids, 7:13] = 0  # Zero velocities

        # Reset episode tracking
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        # Set states in simulation
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def reset(self):
        """Reset all environments"""
        self.reset_idx(torch.arange(self.num_envs, device=self.sim_device))
        self.compute_observations()
        return self.obs_buf

    def compute_observations(self):
        """Compute observations for all environments"""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # Compute base orientation
        base_quat = self.base_quat

        # Convert quaternion to up vector (z-axis in world frame)
        up_vec = quat_rotate(base_quat, torch.tensor([0., 0., 1.], device=self.sim_device).repeat(self.num_envs, 1))

        # Heading vector (forward direction)
        heading_vec = quat_rotate(base_quat,
                                  torch.tensor([1., 0., 0.], device=self.sim_device).repeat(self.num_envs, 1))

        # Build observation vector
        # [base_z, up_vec(3), heading(2), dof_pos(num_dof), dof_vel(num_dof), actions(num_actions)]
        self.obs_buf = torch.cat([
            self.base_pos[:, 2:3],  # Base height (z)
            up_vec,  # Base orientation (up vector)
            heading_vec[:, :2],  # Forward direction (x, y)
            self.dof_pos,  # Joint positions
            self.dof_vel,  # Joint velocities
            self.actions  # Previous actions
        ], dim=-1)

    def compute_reward(self):
        """Compute rewards for all environments"""
        # Forward velocity reward (main objective: walk forward)
        forward_vel = self.base_lin_vel[:, 0]
        forward_reward = forward_vel * self.reward_scales["forwardVel"]

        # Upright reward (stay vertical)
        up_vec = quat_rotate(self.base_quat,
                             torch.tensor([0., 0., 1.], device=self.sim_device).repeat(self.num_envs, 1))
        upright_reward = torch.clamp(up_vec[:, 2], 0, 1) * self.reward_scales["upright"]

        # Height reward (maintain target height)
        target_height = 0.98  # Poppy standing height
        height_error = torch.abs(self.base_pos[:, 2] - target_height)
        height_reward = torch.exp(-10.0 * height_error) * self.reward_scales["height"]

        # Energy penalty (encourage efficient movement)
        energy_penalty = torch.sum(torch.abs(self.actions * self.dof_vel), dim=1) * self.reward_scales["energy"]

        # Action rate penalty (smooth actions)
        action_rate = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        action_rate_penalty = action_rate * self.reward_scales["actionRate"]

        # Total reward
        self.rew_buf[:] = forward_reward + upright_reward + height_reward - energy_penalty - action_rate_penalty

        # Check for termination (fallen down)
        self.reset_buf = torch.where(
            self.base_pos[:, 2] < 0.5,  # Fell down
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )

        # Timeout
        self.reset_buf = torch.where(
            self.progress_buf >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )

    def step(self, actions):
        """Step simulation with actions"""
        # Store previous actions for smoothness penalty
        self.last_actions = self.actions.clone()
        self.actions = actions.clone()

        # Apply actions (torques)
        forces = self.actions * 300.0  # Scale to effort limits
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(forces))

        # Step simulation
        for _ in range(self.control_freq):
            self.gym.simulate(self.sim)
            if not self.headless:
                self.gym.fetch_results(self.sim, True)

        # Update graphics
        if not self.headless:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

        # Update buffers
        self.progress_buf += 1

        # Compute observations and rewards
        self.compute_observations()
        self.compute_reward()

        # Reset environments that are done
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        return self.obs_buf, self.rew_buf, self.reset_buf, {}

    def create_viewer(self):
        """Create viewer for visualization"""
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            cam_pos = gymapi.Vec3(4, 3, 2)
            cam_target = gymapi.Vec3(0, 0, 1)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def __del__(self):
        """Cleanup"""
        if not self.headless and hasattr(self, 'viewer'):
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


def quat_rotate(q, v):
    """Rotate vector by quaternion"""
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c