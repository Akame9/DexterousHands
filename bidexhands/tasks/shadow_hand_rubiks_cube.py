# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from unittest import TextTestRunner
from matplotlib.pyplot import axis
from PIL import Image as Im

import numpy as np
import os
import random
import torch
import math
from bidexhands.utils.torch_jit_utils import *
from bidexhands.tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi


class ShadowHandRubiksCube(BaseTask):
    """
    This class corresponds to the Scissors task. This environment involves two hands and scissors, 
    we need to use two hands to open the scissors

    Args:
        cfg (dict): The configuration file of the environment, which is the parameter defined in the
            dexteroushandenvs/cfg folder

        sim_params (isaacgym._bindings.linux-x86_64.gym_37.SimParams): Isaacgym simulation parameters 
            which contains the parameter settings of the isaacgym physics engine. Also defined in the 
            dexteroushandenvs/cfg folder

        physics_engine (isaacgym._bindings.linux-x86_64.gym_37.SimType): Isaacgym simulation backend
            type, which only contains two members: PhysX and Flex. Our environment use the PhysX backend

        device_type (str): Specify the computing device for isaacgym simulation calculation, there are 
            two options: 'cuda' and 'cpu'. The default is 'cuda'

        device_id (int): Specifies the number of the computing device used when simulating. It is only 
            useful when device_type is cuda. For example, when device_id is 1, the device used 
            is 'cuda:1'

        headless (bool): Specifies whether to visualize during training

        agent_index (list): Specifies how to divide the agents of the hands, useful only when using a 
            multi-agent algorithm. It contains two lists, representing the left hand and the right hand. 
            Each list has six numbers from 0 to 5, representing the palm, middle finger, ring finger, 
            tail finger, index finger, and thumb. Each part can be combined arbitrarily, and if placed 
            in the same list, it means that it is divided into the same agent. The default setting is
            [[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], which means that the two whole hands are 
            regarded as one agent respectively.

        is_multi_agent (bool): Specifies whether it is a multi-agent environment
    """
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.object_type = self.cfg["env"]["objectType"]
        # assert self.object_type in ["block", "egg", "pen"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            # "pot": "mjcf/pot.xml",
            "pot": "mjcf/open_ai_assets/hand/simple_cube.xml"  #"mjcf/open_ai_assets/hand/color_cube.urdf"  #"mjcf/scissors/10495/scissors.xml" #"mjcf/scissors/10495/mobility.urdf"
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["point_cloud", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [point_cloud, full_state]")

        print("Obs type:", self.obs_type)

        self.num_point_cloud_feature_dim = 768
        self.num_obs_dict = {
            "point_cloud": 417 + self.num_point_cloud_feature_dim * 3,
            "point_cloud_for_distill": 417 + self.num_point_cloud_feature_dim * 3,
            "full_state": 553 #502 #417
        }
        self.num_hand_obs = 72 + 95 + 26 + 6
        self.up_axis = 'z'

        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal"]
        self.a_fingertips = ["robot1:ffdistal", "robot1:mfdistal", "robot1:rfdistal", "robot1:lfdistal", "robot1:thdistal"]

        self.hand_center = ["robot1:palm"]

        self.num_fingertips = len(self.fingertips) * 2

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 26
            
        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 52

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        if self.obs_type in ["point_cloud"]:
            from PIL import Image as Im
            from bidexhands.utils import o3dviewer
            # from pointnet2_ops import pointnet2_utils

        self.camera_debug = self.cfg["env"].get("cameraDebug", False)
        self.point_cloud_debug = self.cfg["env"].get("pointCloudDebug", False)

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
    
        #Aathira change
        num_actors = actor_root_state_tensor.shape[0]
        num_rigid_bodies = rigid_body_tensor.shape


        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs * 2 + self.num_object_dofs * 2)
        self.dof_force_tensor = self.dof_force_tensor[:, :48]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        # self.shadow_hand_default_dof_pos = to_torch([0.0, 0.0, -0,  -0,  -0,  -0, -0, -0,
        #                                     -0,  -0, -0,  -0,  -0,  -0, -0, -0,
        #                                     -0,  -0, -0,  -1.04,  1.2,  0., 0, -1.57], dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.shadow_hand_another_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs*2]
        self.shadow_hand_another_dof_pos = self.shadow_hand_another_dof_state[..., 0]
        self.shadow_hand_another_dof_vel = self.shadow_hand_another_dof_state[..., 1]

        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_shadow_hand_dofs*2:self.num_shadow_hand_dofs*2 + self.num_object_dofs]
        self.object_dof_pos = self.object_dof_state[..., 0]
        self.object_dof_vel = self.object_dof_state[..., 1]
        
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        # 
        print("Aathira : self.rigid_body_states.shape = ", self.rigid_body_states.shape)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone() 

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

        self.total_successes = 0
        self.total_resets = 0

    def create_sim(self):
        """
        Allocates which device will simulate and which device will render the scene. Defines the simulation type to be used
        """
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        """
        Adds ground plane to simulation
        """

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def add_cube_colors(self, env_ptr, object_handle):
        color_dict = {
            0: gymapi.Vec3(1.0, 0.0, 0.0),   # Red    ---
            1: gymapi.Vec3(0.0, 1.0, 0.0),   # Green
            2: gymapi.Vec3(0.0, 0.0, 1.0),   # Blue
            3: gymapi.Vec3(1.0, 1.0, 0.0),   # Yellow
            4: gymapi.Vec3(1.0, 0.0, 1.0),   # Magenta
            5: gymapi.Vec3(0.0, 1.0, 1.0),   # Cyan
            6: gymapi.Vec3(0.5, 0.0, 0.0),   # Dark Red      --
            7: gymapi.Vec3(0.0, 0.5, 0.0),   # Dark Green
            8: gymapi.Vec3(0.0, 0.0, 0.5),   # Dark Blue
            9: gymapi.Vec3(0.5, 0.5, 0.0),   # Olive
            10: gymapi.Vec3(0.5, 0.0, 0.5),  # Purple
            11: gymapi.Vec3(0.0, 0.5, 0.5),  # Teal
            12: gymapi.Vec3(0.75, 0.25, 0.0),# Orange
            13: gymapi.Vec3(0.25, 0.75, 0.0),# Lime Green
            14: gymapi.Vec3(0.75, 0.0, 0.25),# Pink
            15: gymapi.Vec3(0.25, 0.0, 0.75),# Indigo
            16: gymapi.Vec3(0.0, 0.25, 0.75),# Sky Blue
            17: gymapi.Vec3(0.75, 0.75, 0.25),# Light Yellow  
            18: gymapi.Vec3(0.25, 0.75, 0.75),# Light Cyan
            19: gymapi.Vec3(0.75, 0.25, 0.75),# Light Magenta
            20: gymapi.Vec3(1.0, 0.5, 0.0),  # Bright Orange
            21: gymapi.Vec3(0.5, 1.0, 0.0),  # Bright Lime Green
            22: gymapi.Vec3(0.0, 1.0, 0.5),  # Bright Teal
            23: gymapi.Vec3(1.0, 0.0, 0.5),  # Bright Pink
            24: gymapi.Vec3(0.5, 0.0, 1.0),  # Bright Indigo
            25: gymapi.Vec3(0.0, 0.5, 1.0),  # Bright Sky Blue
            26: gymapi.Vec3(0.5, 0.5, 0.5)   # Gray
            
        }

        # Set the color for each rigid body using the color dictionary
        for o in range(27):
            color = color_dict[o]
            self.gym.set_rigid_body_color(env_ptr, object_handle, o, gymapi.MESH_VISUAL, color)

    def rotate_rubiks_face(self, dof_states, face_dofs_index, angle):
        """
        Rotates a specific face of the Rubik's cube by setting the target position
        for all DOFs in that face.

        Args:
            gym (gymapi.Gym): The Isaac Gym API instance.
            env_ptr (gymapi.Env): The environment pointer.
            actor_handle (gymapi.Actor): The Rubik's cube actor handle.
            face_dofs (list of str): List of DOF names corresponding to the joints in the face to rotate.
            angle (float): The rotation angle in radians. Default is 90 degrees (π/2).
        """
        # Retrieve the current state of all DOFs
        
        # Loop through each DOF name in the specified face
        for dof_index in face_dofs_index:
            # Get the index of the DOF for this actuator
            
            # Update the position of the DOF for rotation
            dof_states["pos"][dof_index] += angle  # Set target position to the desired angle
            dof_states["vel"][dof_index] = 0      # Set velocity to zero for an instantaneous effect

        # Apply the updated DOF states back to the actor
        return dof_states


    def initialize_goal_faces(self, env_id):
        """
        Initialize and rotate specific faces of the goal cube.

        Args:
            goal_asset: The goal cube asset.
            goal_handle: The handle for the goal cube in the current environment.
            env_ptr: The pointer to the current environment.

        Returns:
            dof_states: The DOF states after applying the rotations.
        """
        env_ptr = self.key_env_ptr.get(env_id)
        goal_handle = self.goal_handles.get(env_id)
        goal_num_dofs = self.gym.get_actor_dof_count(env_ptr, goal_handle)

        # Define face DOFs for right, top, bottom, and left faces
        right_face_dofs = ["pX", "pX_pY_pZ_0", "nZ_pX_pY_0", "nY_pX_pZ_0","nY_nZ_pX_0", "pX_pY_0", "nY_pX_0", "pX_pZ_0","nZ_pX_0"] # Original
        #top_face_dofs = ["pZ", "nX_pZ_2", "pY_pZ_2","nY_pZ_2", "nX_pY_pZ_2","nX_nY_pZ_2", "pX_pZ_2", "pX_pY_pZ_2", "nY_pX_pZ_2"] # Original
        #top_face_dofs = [ "pZ", "nX_pZ_2", "pY_pZ_2","nY_pZ_2", "nX_pY_pZ_2","nX_nY_pZ_2", "pX_pY_2", "pX_pY_pZ_2", "nZ_pX_pY_2"]  #  After right rotation - forward
        #top_face_dofs = ["pZ", "nX_pZ_2", "pY_pZ_2", "nY_pZ_2", "nX_pY_pZ_2", "nX_nY_pZ_2", "pX_pY_2", "pX_pY_pZ_2", "nZ_pX_pY_2"]
        #bottom_face_dofs = ["nZ", "nZ_pX_2", "nZ_pY_2","nY_nZ_2", "nZ_pX_pY_2","nY_nZ_pX_2", "nX_nZ_pY_2", "nX_nY_nZ_2", "nX_nZ_2"] # Original
        #left_face_dofs = ["nX", "nX_pZ_0", "nX_pY_pZ_0", "nX_nY_pZ_0","nX_pY_0", "nX_nZ_pY_0", "nX_nY_0", "nX_nZ_0","nX_nY_nZ_0"] # Original

        # Helper to get DOF indices
        def get_dof_indices(dofs):
            return [dof_index for dof_index in range(goal_num_dofs) if self.gym.get_asset_dof_name(self.goal_asset, dof_index) in dofs]

        # Collect DOF indices for each face
        right_face_dof_index = get_dof_indices(right_face_dofs)
        
        # Get current DOF states and apply rotations
        dof_states = self.gym.get_actor_dof_states(env_ptr, goal_handle, gymapi.STATE_ALL)
        dof_states = self.rotate_rubiks_face(dof_states, right_face_dof_index, np.pi / 2)
        #dof_states= self.rotate_rubiks_face(dof_states, left_face_dof_index, np.pi/2)
        #self.gym.set_actor_dof_states(env_ptr, goal_handle, dof_states, gymapi.STATE_ALL)

        #dof_states = self.rotate_rubiks_face(dof_states, top_face_dof_index, np.pi/6)
        #self.gym.set_actor_dof_states(env_ptr, goal_handle, dof_states, gymapi.STATE_ALL)
        #print("AATHIRA : Dof states after top rotation : ", self.gym.get_actor_dof_states(env_ptr, goal_handle, gymapi.STATE_ALL))
            
        #dof_states= self.rotate_rubiks_face(dof_states, right_face_dof_index, np.pi/2)
        #self.gym.set_actor_dof_states(env_ptr, goal_handle, dof_states, gymapi.STATE_ALL)
        #dof_states= self.rotate_rubiks_face(dof_states, bottom_face_dof_index, np.pi/2)
        #self.gym.set_actor_dof_states(env_ptr, goal_handle, dof_states, gymapi.STATE_ALL)
            
        # Additional rotations can be applied as needed
        self.gym.set_actor_dof_states(env_ptr, goal_handle, dof_states, gymapi.STATE_ALL)
        #print(f"AATHIRA : dof_states {count}:  ", dof_states)
        return dof_states.copy()

    def _create_envs(self, num_envs, spacing, num_per_row):
        """
        Create multiple parallel isaacgym environments

        Args:
            num_envs (int): The total number of environment 

            spacing (float): Specifies half the side length of the square area occupied by each environment

            num_per_row (int): Specify how many environments in a row
        """
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../../assets"
        shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"
        shadow_hand_another_asset_file = "mjcf/open_ai_assets/hand/shadow_hand1.xml" #"mjcf/open_ai_assets/hand/left_hand.xml" 
        table_texture_files = "../assets/textures/texture_stone_stone_texture_0.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            shadow_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file)

        object_asset_file = self.asset_files_dict[self.object_type]

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)
        shadow_hand_another_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_another_asset_file, asset_options)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)

        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)

        # tendon set up
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        a_relevant_tendons = ["robot1:T_FFJ1c", "robot1:T_MFJ1c", "robot1:T_RFJ1c", "robot1:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)
        a_tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_another_asset)

        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
            for rt in a_relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_another_asset, i) == rt:
                    a_tendon_props[i].limit_stiffness = limit_stiffness
                    a_tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)
        self.gym.set_asset_tendon_properties(shadow_hand_another_asset, a_tendon_props)
        
        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names]

        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)
        shadow_hand_another_dof_props = self.gym.get_asset_dof_properties(shadow_hand_another_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        for dof_index in range(self.num_shadow_hand_dofs): 
            shadow_hand_dof_name1 = self.gym.get_asset_dof_name(shadow_hand_asset, dof_index)
            print("shadow_hand_asset : ")
            print(f"{dof_index} : {shadow_hand_dof_name1}")
            
        for dof_index in range(self.num_shadow_hand_dofs): 
            shadow_hand_dof_name2 = self.gym.get_asset_dof_name(shadow_hand_another_asset, dof_index)
            print("shadow_hand_another_asset : ")
            print(f"{dof_index} : {shadow_hand_dof_name2}")

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = False
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # object_asset_options.use_mesh_materials = True
        # object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        # object_asset_options.override_com = True
        # object_asset_options.override_inertia = True
        # object_asset_options.vhacd_enabled = True
        # object_asset_options.vhacd_params = gymapi.VhacdParams()
        # object_asset_options.vhacd_params.resolution = 100000

        
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        self.goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
        
        self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)

        # set object dof properties
        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
        object_dof_props = self.gym.get_asset_dof_properties(object_asset)

        self.object_dof_lower_limits = []
        self.object_dof_upper_limits = []

        for i in range(self.num_object_dofs):
            self.object_dof_lower_limits.append(object_dof_props['lower'][i])
            self.object_dof_upper_limits.append(object_dof_props['upper'][i])

        self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
        self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)

        # create table asset
        table_dims = gymapi.Vec3(0.5, 1.0, 0.6)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, gymapi.AssetOptions())
        
        """
        object_start_pose.p.x = shadow_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz
        """
        shadow_hand_start_pose = gymapi.Transform()
        #shadow_hand_start_pose.p = gymapi.Vec3(0.55, 0.5, 0.8)
        shadow_hand_start_pose.p = gymapi.Vec3(0.35, 0.15, 0.8)
        # AATHIRA : Turn hand
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(3.14159, 0+1.57, 1.57)

        shadow_another_hand_start_pose = gymapi.Transform()
        #shadow_hand_start_pose.p = gymapi.Vec3(0.55, 0.5, 0.8)
        shadow_another_hand_start_pose.p = gymapi.Vec3(0.35, -0.15, 0.8)
        shadow_another_hand_start_pose.r = gymapi.Quat().from_euler_zyx(3.14159, 0-1.57, 1.57)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0., 0.6)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 1.57)
        pose_dx, pose_dy, pose_dz = -1.0, 0.0, -0.0
       
        # Aathira changed
        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.3)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        goal_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 1.57)  
        
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies * 2 + 2 * self.num_object_bodies + 1
        max_agg_shapes = self.num_shadow_hand_shapes * 2 + 2 * self.num_object_shapes + 1

        self.shadow_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.another_hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []
        self.bucket_indices = []
        self.ball_indices = []

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]
        self.fingertip_another_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_another_asset, name) for name in self.a_fingertips]

        # create fingertip force sensors, if needed
        sensor_pose = gymapi.Transform()
        for ft_handle in self.fingertip_handles:
            self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)
        for ft_a_handle in self.fingertip_another_handles:
            self.gym.create_asset_force_sensor(shadow_hand_another_asset, ft_a_handle, sensor_pose)

        #Aathira changes
        self.goal_dof_states = []
        self.key_env_ptr = {} 
        self.object_handles = {}
        self.goal_handles = {}
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            self.key_env_ptr[i] = env_ptr

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)


            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, 0, 0)
            shadow_hand_another_actor = self.gym.create_actor(env_ptr, shadow_hand_another_asset, shadow_another_hand_start_pose, "another_hand", i, 0, 0)
            
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_another_actor, shadow_hand_another_dof_props)
            another_hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_another_actor, gymapi.DOMAIN_SIM)
            self.another_hand_indices.append(another_hand_idx)            

            # randomize colors and textures for rigid body
            num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, shadow_hand_actor)
            #print("AATHIRA : Hand rigid bodies count", num_bodies)
            #print("AATHIRA : Another Hand rigid bodies count", self.gym.get_actor_rigid_body_count(env_ptr, shadow_hand_another_actor))
            hand_props = self.gym.get_actor_rigid_body_properties(env_ptr, shadow_hand_actor)
            #print("AATHIRA : Hand pros shape = ", len(hand_props))
            hand_rigid_body_index = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]
            
            for n in self.agent_index[0]:
                colorx = random.uniform(0, 1)
                colory = random.uniform(0, 1)
                colorz = random.uniform(0, 1)
                for m in n:
                    for o in hand_rigid_body_index[m]:
                        self.gym.set_rigid_body_color(env_ptr, shadow_hand_actor, o, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(colorx, colory, colorz))
            for n in self.agent_index[1]:                
                colorx = random.uniform(0, 1)
                colory = random.uniform(0, 1)
                colorz = random.uniform(0, 1)
                for m in n:
                    for o in hand_rigid_body_index[m]:
                        self.gym.set_rigid_body_color(env_ptr, shadow_hand_another_actor, o, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(colorx, colory, colorz))
               
            # create fingertip force-torque sensors
            self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)
            self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_another_actor)
            
            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.object_handles[i] = object_handle
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, object_handle, object_dof_props)
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            #print("AATHIRA : OBJECT_IDX : ",object_idx)
            self.object_indices.append(object_idx)
            #cube_rigid_bodies = self.gym.get_actor_rigid_body_count(env_ptr, object_handle)
            #print("AATHIRA : cube rigid bodies", cube_rigid_bodies)
            #cube_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            #print("AATHIRA : Cube pros shape = ", len(cube_props))
            # self.gym.set_actor_scale(env_ptr, object_handle, 0.3)
            self.add_cube_colors(env_ptr, object_handle)
            

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, self.goal_asset, goal_start_pose, "goal_object", i+self.num_envs , 0, 0) 
            self.goal_handles[i] = goal_handle
            #self.gym.set_actor_dof_properties(env_ptr, goal_handle, object_dof_props)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)
            self.add_cube_colors(env_ptr, goal_handle)

            # NOTE : get_actor_dof_states is being called inside initialize_goal_faces()
            goal_dof_state = self.initialize_goal_faces(i)
            self.goal_dof_states.append(goal_dof_state)
            

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0, 0)
            self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)
            
            
            """
            AATHIRA : Hand rigid bodies count 26
            AATHIRA : Another Hand rigid bodies count 26
            AATHIRA : Hand pros shape =  26
            AATHIRA : OBJECT_IDX :  2
            AATHIRA : cube rigid bodies 27
            AATHIRA : Cube pros shape =  27
            AATHIRA : Goal Index  3
            AATHIRA : Goal rigid bodies count 27
            AATHIRA : Table rigid bodies count 1
            Aathira : Number of actors: 5
            Aathira : Number of rigid bodies: (107, 13)
            Aathira : self.rigid_body_states.shape =  torch.Size([1, 107, 13])
            """
            
            #set friction
            #another_hand_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, shadow_hand_another_actor)
            #object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            #another_hand_shape_props[0].friction = 1
            #object_shape_props[0].friction = 1
            #self.gym.set_actor_rigid_shape_properties(env_ptr, shadow_hand_another_actor, another_hand_shape_props)
            #self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        
        self.goal_init_state = self.goal_states.clone()
        
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.fingertip_another_handles = to_torch(self.fingertip_another_handles, dtype=torch.long, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.another_hand_indices = to_torch(self.another_hand_indices, dtype=torch.long, device=self.device)

        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)
        self.bucket_indices = to_torch(self.bucket_indices, dtype=torch.long, device=self.device)
        self.ball_indices = to_torch(self.ball_indices, dtype=torch.long, device=self.device)
        self.goal_dof_states = to_torch(self.goal_dof_states, dtype=torch.long, device=self.device)
        
    def compute_reward(self, actions):
        """
        Compute the reward of all environment. The core function is compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.scissors_right_handle_pos, self.scissors_left_handle_pos, self.object_dof_pos,
            self.left_hand_pos, self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos, 
            self.left_hand_ff_pos, self.left_hand_mf_pos, self.left_hand_rf_pos, self.left_hand_lf_pos, self.left_hand_th_pos, 
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )
        , which we will introduce in detail there

        Args:
            actions (tensor): Actions of agents in the all environment 
        """
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.object_dof_pos,
            self.left_hand_pos, self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos, 
            self.left_hand_ff_pos, self.left_hand_mf_pos, self.left_hand_rf_pos, self.left_hand_lf_pos, self.left_hand_th_pos, 
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen"), self.num_envs, self.object_dof_state, self.goal_dof_states
        )

        """
        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes
        
        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))
        """
    def compute_observations(self):
        """
        Compute the observations of all environment. The core function is self.compute_full_state(True), 
        which we will introduce in detail there

        """
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        if self.obs_type in ["point_cloud"]:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]
        
        #AATHIRA
        #obj_dof_states = []
        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_shadow_hand_dofs*2:self.num_shadow_hand_dofs*2 + self.num_object_dofs]

        #for i in range(self.num_envs):
        #    object_handle = self.object_handles.get(i)
        #    env_ptr = self.key_env_ptr.get(i)
        #    dof_states = self.gym.get_actor_dof_states(env_ptr, object_handle, gymapi.STATE_ALL)
        #    obj_dof_states.append(dof_states.copy())
        #self.object_dof_states = obj_dof_states # convert into tensor

        #Aathira : For debug lines to see the axis
        self.cube_1_pos = self.rigid_body_states[:, 26 * 2 + 1, 0:3]
        self.cube_1_rot = self.rigid_body_states[:, 26 * 2 + 1, 3:7]

        self.left_hand_pos = self.rigid_body_states[:, 3 + 26, 0:3]
        self.left_hand_rot = self.rigid_body_states[:, 3 + 26, 3:7]
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        self.right_hand_pos = self.rigid_body_states[:, 3, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, 3, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # right hand finger
        self.right_hand_ff_pos = self.rigid_body_states[:, 7, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, 7, 3:7]
        self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_mf_pos = self.rigid_body_states[:, 11, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, 11, 3:7]
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_rf_pos = self.rigid_body_states[:, 15, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, 15, 3:7]
        self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_lf_pos = self.rigid_body_states[:, 20, 0:3]
        self.right_hand_lf_rot = self.rigid_body_states[:, 20, 3:7]
        self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_th_pos = self.rigid_body_states[:, 25, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, 25, 3:7]
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.left_hand_ff_pos = self.rigid_body_states[:, 7 + 26, 0:3]
        self.left_hand_ff_rot = self.rigid_body_states[:, 7 + 26, 3:7]
        self.left_hand_ff_pos = self.left_hand_ff_pos + quat_apply(self.left_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_mf_pos = self.rigid_body_states[:, 11 + 26, 0:3]
        self.left_hand_mf_rot = self.rigid_body_states[:, 11 + 26, 3:7]
        self.left_hand_mf_pos = self.left_hand_mf_pos + quat_apply(self.left_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_rf_pos = self.rigid_body_states[:, 15 + 26, 0:3]
        self.left_hand_rf_rot = self.rigid_body_states[:, 15 + 26, 3:7]
        self.left_hand_rf_pos = self.left_hand_rf_pos + quat_apply(self.left_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_lf_pos = self.rigid_body_states[:, 20 + 26, 0:3]
        self.left_hand_lf_rot = self.rigid_body_states[:, 20 + 26, 3:7]
        self.left_hand_lf_pos = self.left_hand_lf_pos + quat_apply(self.left_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_th_pos = self.rigid_body_states[:, 25 + 26, 0:3]
        self.left_hand_th_rot = self.rigid_body_states[:, 25 + 26, 3:7]
        self.left_hand_th_pos = self.left_hand_th_pos + quat_apply(self.left_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]
        self.fingertip_another_state = self.rigid_body_states[:, self.fingertip_another_handles][:, :, 0:13]
        self.fingertip_another_pos = self.rigid_body_states[:, self.fingertip_another_handles][:, :, 0:3]

        if self.obs_type == "full_state":
            self.compute_full_state()
        elif self.obs_type == "point_cloud":
            self.compute_point_cloud_observation()

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_state(self, asymm_obs=False):
        """
        Compute the observations of all environment. The observation is composed of three parts: 
        the state values of the left and right hands, and the information of objects and target. 
        The state values of the left and right hands were the same for each task, including hand 
        joint and finger positions, velocity, and force information. The detail 428-dimensional 
        observational space as shown in below:

        Index       Description
        0 - 23	    right shadow hand dof position
        24 - 47	    right shadow hand dof velocity
        48 - 71	    right shadow hand dof force
        72 - 136	right shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
        137 - 166	right shadow hand fingertip force, torque (5 x 6)
        167 - 169	right shadow hand base position
        170 - 172	right shadow hand base rotation
        173 - 198	right shadow hand actions
        199 - 222	left shadow hand dof position
        223 - 246	left shadow hand dof velocity
        247 - 270	left shadow hand dof force
        271 - 335	left shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
        336 - 365	left shadow hand fingertip force, torque (5 x 6)
        366 - 368	left shadow hand base position
        369 - 371	left shadow hand base rotation
        372 - 397	left shadow hand actions
        398 - 404	object pose
        405 - 407	object linear velocity
        408 - 410	object angle velocity
        411 - 417	goal pose
        418 - 421	goal rot - object rot
        422 - 553   cube positions
        #422 - 424	scissors right handle position
        #425 - 427	scissors left handle position
        """
        num_ft_states = 13 * int(self.num_fingertips / 2)  # 65
        num_ft_force_torques = 6 * int(self.num_fingertips / 2)  # 30

        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                            self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]

        fingertip_obs_start = 72  # 168 = 157 + 11
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
                    num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]
        
        hand_pose_start = fingertip_obs_start + 95
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.right_hand_pos
        self.obs_buf[:, hand_pose_start+3:hand_pose_start+4] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+4:hand_pose_start+5] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+5:hand_pose_start+6] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        self.obs_buf[:, action_obs_start:action_obs_start + 26] = self.actions[:, :26]

        # another_hand
        another_hand_start = action_obs_start + 26
        self.obs_buf[:, another_hand_start:self.num_shadow_hand_dofs + another_hand_start] = unscale(self.shadow_hand_another_dof_pos,
                                                            self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, self.num_shadow_hand_dofs + another_hand_start:2*self.num_shadow_hand_dofs + another_hand_start] = self.vel_obs_scale * self.shadow_hand_another_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs + another_hand_start:3*self.num_shadow_hand_dofs + another_hand_start] = self.force_torque_obs_scale * self.dof_force_tensor[:, 24:48]

        fingertip_another_obs_start = another_hand_start + 72
        self.obs_buf[:, fingertip_another_obs_start:fingertip_another_obs_start + num_ft_states] = self.fingertip_another_state.reshape(self.num_envs, num_ft_states)
        self.obs_buf[:, fingertip_another_obs_start + num_ft_states:fingertip_another_obs_start + num_ft_states +
                    num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, 30:]

        hand_another_pose_start = fingertip_another_obs_start + 95
        self.obs_buf[:, hand_another_pose_start:hand_another_pose_start + 3] = self.left_hand_pos
        self.obs_buf[:, hand_another_pose_start+3:hand_another_pose_start+4] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_another_pose_start+4:hand_another_pose_start+5] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_another_pose_start+5:hand_another_pose_start+6] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[2].unsqueeze(-1)

        action_another_obs_start = hand_another_pose_start + 6
        self.obs_buf[:, action_another_obs_start:action_another_obs_start + 26] = self.actions[:, 26:]

        obj_obs_start = action_another_obs_start + 26  # 144
        self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel
        
        #cubes_positions = [self.cube_1_pos, self.cube_2_pos, self.cube_3_pos, self.cube_4_pos, self.cube_5_pos, self.cube_6_pos, self.cube_7_pos, self.cube_8_pos,
        #                   self.cube_9_pos, self.cube_10_pos, self.cube_11_pos, self.cube_12_pos, self.cube_13_pos, self.cube_14_pos, self.cube_15_pos, self.cube_16_pos, 
        #                   self.cube_17_pos, self.cube_18_pos, self.cube_19_pos, self.cube_20_pos, self.cube_21_pos, self.cube_22_pos, self.cube_23_pos, self.cube_24_pos, 
        #                   self.cube_25_pos, self.cube_26_pos, self.cube_27_pos]  # List of all cube position tensors
        
        #self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 94] = torch.cat(cubes_positions, dim=1)
        object_dof_states_start = obj_obs_start + 13
        num_object_dof_states = self.object_dof_state.shape[1] * self.object_dof_state.shape[2]
        self.obs_buf[:, object_dof_states_start:object_dof_states_start + num_object_dof_states] = self.object_dof_state.reshape(self.num_envs, num_object_dof_states)

    def reset_target_pose(self, env_ids, apply_reset=False):
        """
        Reset and randomize the goal pose

        Args:
            env_ids (tensor): The index of the environment that needs to reset goal pose

            apply_reset (bool): Whether to reset the goal directly here, usually used 
            when the same task wants to complete multiple goals

        """
        #Aathira
        # Why do we need to reset goal?
        # Reset the goal cube using the rotate_face function.
        # Ignore reset_goal_buf not being used
        
        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        #self.goal_states[env_ids, 2] += 10.0

        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor #self.goal_init_state[env_ids, 0:3]
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7] #self.goal_init_state[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        #dof_states = self.initialize_goal_faces(env_ids)

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        
        self.reset_goal_buf[env_ids] = 0

    def reset(self, env_ids, goal_env_ids):
        """
        Reset and randomize the environment

        Args:
            env_ids (tensor): The index of the environment that needs to reset

            goal_env_ids (tensor): The index of the environment that only goals need reset

        """

        # Set target pose for the objects to their default positions
        self.reset_target_pose(env_ids)

        # Reset object state to initial values without randomization
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()

        # Reset object orientation to initial orientation
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = self.object_init_state[env_ids, 3:7]

        # Set object velocities to zero
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        # Handle goal indices for object resets
        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                self.goal_object_indices[env_ids],
                                                self.goal_object_indices[goal_env_ids]]).to(torch.int32))

        # Reset shadow hand DOF positions to default without noise
        pos = self.shadow_hand_default_dof_pos
        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_another_dof_pos[env_ids, :] = pos

        # Reset shadow hand DOF velocities to default without noise
        vel = self.shadow_hand_dof_default_vel
        self.shadow_hand_dof_vel[env_ids, :] = vel
        self.shadow_hand_another_dof_vel[env_ids, :] = vel

        # Initialize targets to the default positions
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.prev_targets[env_ids, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs*2] = pos
        self.cur_targets[env_ids, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs*2] = pos

        # Set indexed tensors in the simulator
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        another_hand_indices = self.another_hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(torch.cat([hand_indices, another_hand_indices]).to(torch.int32))

        # Set DOF position targets for hands
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(all_hand_indices),
            len(all_hand_indices)
        )

        all_hand_and_object_indices = torch.unique(torch.cat([all_hand_indices,self.object_indices[env_ids]]).to(torch.int32))
        print("Lenght of self.dof_states : ", len(self.dof_state))
        print("Length of all_hand_indices : ", len(all_hand_and_object_indices))
        
    
        # Set DOF states and actor root states for hands and objects
        # For goal we are already setting in the initialization function called inside reset_target_pose()
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(all_hand_and_object_indices),
            len(all_hand_and_object_indices)
        )

        all_indices = torch.unique(torch.cat([all_hand_indices, object_indices]).to(torch.int32))
        print("Length of all_indices : ", len(all_indices))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(all_indices),
            len(all_indices)
        )

        # Reset progress, reset buffer, and successes for the environments
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0


    def pre_physics_step(self, actions):
        """
        The pre-processing of the physics step. Determine whether the reset environment is needed, 
        and calculate the next movement of Shadowhand through the given action. The 52-dimensional 
        action space as shown in below:
        
        Index   Description
        0 - 19 	right shadow hand actuated joint
        20 - 22	right shadow hand base translation
        23 - 25	right shadow hand base rotation
        26 - 45	left shadow hand actuated joint
        46 - 48	left shadow hand base translation
        49 - 51	left shadow hand base rotatio

        Args:
            actions (tensor): Actions of agents in the all environment 
        """
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            print("AATHIRA Inside pre_physics_step env_ids : ", env_ids)
            #Uncomment
            #self.reset(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 6:26],
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

            self.cur_targets[:, self.actuated_dof_indices + 24] = scale(self.actions[:, 32:52],
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            
            self.cur_targets[:, self.actuated_dof_indices + 24] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices + 24] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            
            self.cur_targets[:, self.actuated_dof_indices + 24] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices + 24],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            # self.cur_targets[:, 49] = scale(self.actions[:, 0],
            #                                 self.object_dof_lower_limits[0], self.object_dof_upper_limits[0])

            self.apply_forces[:, 1, :] = actions[:, 0:3] * self.dt * self.transition_scale * 100000
            self.apply_forces[:, 1 + 26, :] = actions[:, 26:29] * self.dt * self.transition_scale * 100000
            self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000
            self.apply_torque[:, 1 + 26, :] = self.actions[:, 29:32] * self.dt * self.orientation_scale * 1000   

            # Uncomment
            #self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces), gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.prev_targets[:, self.actuated_dof_indices + 24] = self.cur_targets[:, self.actuated_dof_indices + 24]
        #Uncomment
        #self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))  

    def post_physics_step(self):
        """
        The post-processing of the physics step. Compute the observation and reward, and visualize auxiliary 
        lines for debug when needed
        
        """
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                #Aathira changes
                self.add_debug_lines(self.envs[i], self.cube_1_pos[i], self.cube_1_rot[i])
                

                self.add_debug_lines(self.envs[i], self.right_hand_ff_pos[i], self.right_hand_ff_rot[i])
                self.add_debug_lines(self.envs[i], self.right_hand_mf_pos[i], self.right_hand_mf_rot[i])
                self.add_debug_lines(self.envs[i], self.right_hand_rf_pos[i], self.right_hand_rf_rot[i])
                self.add_debug_lines(self.envs[i], self.right_hand_lf_pos[i], self.right_hand_lf_rot[i])
                self.add_debug_lines(self.envs[i], self.right_hand_th_pos[i], self.right_hand_th_rot[i])

                #self.add_debug_lines(self.envs[i], self.left_hand_ff_pos[i], self.right_hand_ff_rot[i])
                #self.add_debug_lines(self.envs[i], self.left_hand_mf_pos[i], self.right_hand_mf_rot[i])
                #self.add_debug_lines(self.envs[i], self.left_hand_rf_pos[i], self.right_hand_rf_rot[i])
                #self.add_debug_lines(self.envs[i], self.left_hand_lf_pos[i], self.right_hand_lf_rot[i])
                #self.add_debug_lines(self.envs[i], self.left_hand_th_pos[i], self.right_hand_th_rot[i])


    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()

        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])
     
#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot, object_dof_pos,
    left_hand_pos, right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
    left_hand_ff_pos, left_hand_mf_pos, left_hand_rf_pos, left_hand_lf_pos, left_hand_th_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool,
    num_envs: int, object_dof_states, goal_dof_states
):
    """
    Compute the reward of all environment.

    Args:
        rew_buf (tensor): The reward buffer of all environments at this time

        reset_buf (tensor): The reset buffer of all environments at this time

        reset_goal_buf (tensor): The only-goal reset buffer of all environments at this time

        progress_buf (tensor): The porgress buffer of all environments at this time

        successes (tensor): The successes buffer of all environments at this time

        consecutive_successes (tensor): The consecutive successes buffer of all environments at this time

        max_episode_length (float): The max episode length in this environment

        object_pos (tensor): The position of the object

        object_rot (tensor): The rotation of the object

        target_pos (tensor): The position of the target

        target_rot (tensor): The rotate of the target

        scissors_left_handle_pos (tensor): The position of the left handle of the scissors

        scissors_right_handle_pos (tensor): The position of the right handle of the scissors

        left_hand_pos, right_hand_pos (tensor): The position of the bimanual hands

        right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos (tensor): The position of the five fingers 
            of the right hand

        left_hand_ff_pos, left_hand_mf_pos, left_hand_rf_pos, left_hand_lf_pos, left_hand_th_pos (tensor): The position of the five fingers 
            of the left hand

        dist_reward_scale (float): The scale of the distance reward

        rot_reward_scale (float): The scale of the rotation reward

        rot_eps (float): The epsilon of the rotation calculate

        actions (tensor): The action buffer of all environments at this time

        action_penalty_scale (float): The scale of the action penalty reward

        success_tolerance (float): The tolerance of the success determined

        reach_goal_bonus (float): The reward given when the object reaches the goal

        fall_dist (float): When the object is far from the Shadowhand, it is judged as falling

        fall_penalty (float): The reward given when the object is fell

        max_consecutive_successes (float): The maximum of the consecutive successes

        av_factor (float): The average factor for calculate the consecutive successes

        ignore_z_rot (bool): Is it necessary to ignore the rot of the z-axis, which is usually used 
            for some specific objects (e.g. pen)
    """
    #Aathira : 
    # Assume the initial position of task is two hands grabbing the cube in air with the fingers.

    # Calculate the difference between the position of the object and goal.
    # Calculate the difference between the rotation of the object and goal.
    # Calculate the distance between the right and left fingers to the center of the cube if it's greater than a threshold then add penalty to the reward.
    # Find out which envs hit the goal and update success count and reset the environment(update the reset buffer)
    # Success bonus: orientation is within `success_tolerance` of goal orientation
    # Fall penalty: if the cube falls on the table then reset the environment(update the reset buffer) - decide whether to add a penalty or not to the reward?
    # Check if progress_buf is greater than maximumn episode length then reset the environment(update the reset buffer).
    # Ignore calculating goal_resets, successes, consecutive_successes for now. 
    

    """
    Compute the reward of all environments in this function. Various components of the reward function
    are calculated based on the positions and orientations of the objects, hands, and targets.
    """

    # Calculate distance reward (distance between object and target)
    dist = torch.norm(object_pos - target_pos, dim=-1)
    dist_rew = dist_reward_scale / (dist + 0.1)
    
    # Calculate rotation reward (orientation difference between object and target)
    rot_dist = torch.norm(object_rot - target_rot, dim=-1)
    rot_rew = rot_reward_scale / (rot_dist + rot_eps)

    # Reward based on hand proximity to the cube object
    right_hand_dist = torch.norm(right_hand_pos - object_pos, dim=-1)
    left_hand_dist = torch.norm(left_hand_pos - object_pos, dim=-1)
    hand_rew = -0.5 * (right_hand_dist + left_hand_dist)


    # Calculate DOF distance reward (distance between object DOF states and goal DOF states)
    num_object_dof_states = object_dof_states.shape[1] * object_dof_states.shape[2]
    object_dof_states_reshaped = object_dof_states.reshape(num_envs, num_object_dof_states)
    goal_dof_states_reshaped = goal_dof_states.reshape(num_envs, num_object_dof_states)
    dof_dist = torch.norm(object_dof_states_reshaped - goal_dof_states_reshaped, dim=-1)
    dof_rew = 20 / (dof_dist + 0.1)

    # Success criteria for reaching the goal
    goal_reached = (dist < success_tolerance) & (rot_dist < rot_eps) & (dof_dist < 0.1) # Add cubelet pos and rot
    goal_rew = goal_reached.float() * reach_goal_bonus
    # Action penalty for stabilizing actions
    #action_penalty = torch.sum(actions ** 2, dim=-1)
    #action_penalty_rew = -action_penalty_scale * action_penalty

    # Apply all components to compute the final reward
    rew_buf[:] = dist_rew + rot_rew + goal_rew + hand_rew + dof_rew #+ action_penalty_rew

    # Check if object fell out of reach and apply fall penalty
    fell = (right_hand_dist > fall_dist) | (left_hand_dist > fall_dist)
    fall_rew = fell.float() * fall_penalty
    rew_buf += fall_rew

    # Reset environment if max episode length reached or object fell
    print(f"AATHIRA : (progress_buf >= max_episode_length): {(progress_buf >= max_episode_length)}")
    print(f"AATHIRA : fell : {fell}")
    #print(f"AATHIRA : goal_reached : {goal_reached}")
    resets = (progress_buf >= max_episode_length) | fell | goal_reached
    reset_buf[:] = resets.float()

    # Track successes and apply resets for goal-reaching environments
    #successes += goal_reached.float()
    #goal_resets = goal_reached & (successes >= max_consecutive_successes)
    #reset_goal_buf[:] = goal_resets.float()
    #reset_goal_buf = torch.zeros_like(resets)
    return rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot


"""
======================================================================================================================================================
Index	Abbreviation	Full Form
    0	WRJ1	Wrist Joint 1
    1	WRJ0	Wrist Joint 0
    2	FFJ3	Fore Finger Knuckle Joint
    3	FFJ2	Fore Finger Proximal Joint
    4	FFJ1	Fore Finger Middle Joint
    5	FFJ0	Fore Finger Distal Joint
    6	MFJ3	Middle Finger Knuckle Joint
    7	MFJ2	Middle Finger Proximal Joint
    8	MFJ1	Middle Finger Middle Joint
    9	MFJ0	Middle Finger Distal Joint
    10	RFJ3	Ring Finger Knuckle Joint
    11	RFJ2	Ring Finger Proximal Joint
    12	RFJ1	Ring Finger Middle Joint
    13	RFJ0	Ring Finger Distal Joint
    14	LFJ4	Little Finger Metacarpal Joint
    15	LFJ3	Little Finger Knuckle Joint
    16	LFJ2	Little Finger Proximal Joint
    17	LFJ1	Little Finger Middle Joint
    18	LFJ0	Little Finger Distal Joint
    19	THJ4	Thumb Base Joint
    20	THJ3	Thumb Proximal Joint
    21	THJ2	Thumb Hub Joint
    22	THJ1	Thumb Middle Joint
    23	THJ0	Thumb Distal Joint

shadow_hand_asset : 
0 : robot0:WRJ1
shadow_hand_asset : 
1 : robot0:WRJ0
shadow_hand_asset : 
2 : robot0:FFJ3
shadow_hand_asset : 
3 : robot0:FFJ2
shadow_hand_asset : 
4 : robot0:FFJ1
shadow_hand_asset : 
5 : robot0:FFJ0
shadow_hand_asset : 
6 : robot0:MFJ3
shadow_hand_asset : 
7 : robot0:MFJ2
shadow_hand_asset : 
8 : robot0:MFJ1
shadow_hand_asset : 
9 : robot0:MFJ0
shadow_hand_asset : 
10 : robot0:RFJ3
shadow_hand_asset : 
11 : robot0:RFJ2
shadow_hand_asset : 
12 : robot0:RFJ1
shadow_hand_asset : 
13 : robot0:RFJ0
shadow_hand_asset : 
14 : robot0:LFJ4
shadow_hand_asset : 
15 : robot0:LFJ3
shadow_hand_asset : 
16 : robot0:LFJ2
shadow_hand_asset : 
17 : robot0:LFJ1
shadow_hand_asset : 
18 : robot0:LFJ0
shadow_hand_asset : 
19 : robot0:THJ4
shadow_hand_asset : 
20 : robot0:THJ3
shadow_hand_asset : 
21 : robot0:THJ2
shadow_hand_asset : 
22 : robot0:THJ1
shadow_hand_asset : 
23 : robot0:THJ0
shadow_hand_another_asset : 
0 : robot1:WRJ1
shadow_hand_another_asset : 
1 : robot1:WRJ0
shadow_hand_another_asset : 
2 : robot1:FFJ3
shadow_hand_another_asset : 
3 : robot1:FFJ2
shadow_hand_another_asset : 
4 : robot1:FFJ1
shadow_hand_another_asset : 
5 : robot1:FFJ0
shadow_hand_another_asset : 
6 : robot1:MFJ3
shadow_hand_another_asset : 
7 : robot1:MFJ2
shadow_hand_another_asset : 
8 : robot1:MFJ1
shadow_hand_another_asset : 
9 : robot1:MFJ0
shadow_hand_another_asset : 
10 : robot1:RFJ3
shadow_hand_another_asset : 
11 : robot1:RFJ2
shadow_hand_another_asset : 
12 : robot1:RFJ1
shadow_hand_another_asset : 
13 : robot1:RFJ0
shadow_hand_another_asset : 
14 : robot1:LFJ4
shadow_hand_another_asset : 
15 : robot1:LFJ3
shadow_hand_another_asset : 
16 : robot1:LFJ2
shadow_hand_another_asset : 
17 : robot1:LFJ1
shadow_hand_another_asset : 
18 : robot1:LFJ0
shadow_hand_another_asset : 
19 : robot1:THJ4
shadow_hand_another_asset : 
20 : robot1:THJ3
shadow_hand_another_asset : 
21 : robot1:THJ2
shadow_hand_another_asset : 
22 : robot1:THJ1
shadow_hand_another_asset : 
23 : robot1:THJ0
====================================================================================================================================
        self.cube_1_pos = self.rigid_body_states[:, 26 * 2 + 1, 0:3]
        self.cube_1_rot = self.rigid_body_states[:, 26 * 2 + 1, 3:7] 
        #self.cube_1_pos = self.cube_1_pos + quat_apply(self.cube_1_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.)
        #self.cube_1_pos = self.cube_1_pos + quat_apply(self.cube_1_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.2)
        #self.cube_1_pos = self.cube_1_pos + quat_apply(self.cube_1_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.1)
        self.cube_2_pos = self.rigid_body_states[:, 26 * 2 + 2, 0:3]
        self.cube_2_rot = self.rigid_body_states[:, 26 * 2 + 2, 3:7]
        self.cube_3_pos = self.rigid_body_states[:, 26 * 2 + 3, 0:3]
        self.cube_3_rot = self.rigid_body_states[:, 26 * 2 + 3, 3:7]
        self.cube_4_pos = self.rigid_body_states[:, 26 * 2 + 4, 0:3]
        self.cube_4_rot = self.rigid_body_states[:, 26 * 2 + 4, 3:7]
        self.cube_5_pos = self.rigid_body_states[:, 26 * 2 + 5, 0:3]
        self.cube_5_rot = self.rigid_body_states[:, 26 * 2 + 5, 3:7]
        self.cube_6_pos = self.rigid_body_states[:, 26 * 2 + 6, 0:3]
        self.cube_6_rot = self.rigid_body_states[:, 26 * 2 + 6, 3:7]
        self.cube_7_pos = self.rigid_body_states[:, 26 * 2 + 7, 0:3]
        self.cube_7_rot = self.rigid_body_states[:, 26 * 2 + 7, 3:7]
        self.cube_8_pos = self.rigid_body_states[:, 26 * 2 + 8, 0:3]
        self.cube_8_rot = self.rigid_body_states[:, 26 * 2 + 8, 3:7]
        self.cube_9_pos = self.rigid_body_states[:, 26 * 2 + 9, 0:3]
        self.cube_9_rot = self.rigid_body_states[:, 26 * 2 + 9, 3:7]
        self.cube_10_pos = self.rigid_body_states[:, 26 * 2 + 10, 0:3]
        self.cube_10_rot = self.rigid_body_states[:, 26 * 2 + 10, 3:7]
        self.cube_11_pos = self.rigid_body_states[:, 26 * 2 + 11, 0:3]
        self.cube_11_rot = self.rigid_body_states[:, 26 * 2 + 11, 3:7]
        self.cube_12_pos = self.rigid_body_states[:, 26 * 2 + 12, 0:3]
        self.cube_12_rot = self.rigid_body_states[:, 26 * 2 + 12, 3:7]
        self.cube_13_pos = self.rigid_body_states[:, 26 * 2 + 13, 0:3]
        self.cube_13_rot = self.rigid_body_states[:, 26 * 2 + 13, 3:7]
        self.cube_14_pos = self.rigid_body_states[:, 26 * 2 + 14, 0:3]
        self.cube_14_rot = self.rigid_body_states[:, 26 * 2 + 14, 3:7]
        self.cube_15_pos = self.rigid_body_states[:, 26 * 2 + 15, 0:3]
        self.cube_15_rot = self.rigid_body_states[:, 26 * 2 + 15, 3:7]
        self.cube_16_pos = self.rigid_body_states[:, 26 * 2 + 16, 0:3]
        self.cube_16_rot = self.rigid_body_states[:, 26 * 2 + 16, 3:7]
        self.cube_17_pos = self.rigid_body_states[:, 26 * 2 + 17, 0:3]
        self.cube_17_rot = self.rigid_body_states[:, 26 * 2 + 17, 3:7]
        self.cube_18_pos = self.rigid_body_states[:, 26 * 2 + 18, 0:3]
        self.cube_18_rot = self.rigid_body_states[:, 26 * 2 + 18, 3:7]
        self.cube_19_pos = self.rigid_body_states[:, 26 * 2 + 19, 0:3]
        self.cube_19_rot = self.rigid_body_states[:, 26 * 2 + 19, 3:7]
        self.cube_20_pos = self.rigid_body_states[:, 26 * 2 + 20, 0:3]
        self.cube_20_rot = self.rigid_body_states[:, 26 * 2 + 20, 3:7]
        self.cube_21_pos = self.rigid_body_states[:, 26 * 2 + 21, 0:3]
        self.cube_21_rot = self.rigid_body_states[:, 26 * 2 + 21, 3:7]
        self.cube_22_pos = self.rigid_body_states[:, 26 * 2 + 22, 0:3]
        self.cube_22_rot = self.rigid_body_states[:, 26 * 2 + 22, 3:7]
        self.cube_23_pos = self.rigid_body_states[:, 26 * 2 + 23, 0:3]
        self.cube_23_rot = self.rigid_body_states[:, 26 * 2 + 23, 3:7]
        self.cube_24_pos = self.rigid_body_states[:, 26 * 2 + 24, 0:3]
        self.cube_24_rot = self.rigid_body_states[:, 26 * 2 + 24, 3:7]
        self.cube_25_pos = self.rigid_body_states[:, 26 * 2 + 25, 0:3]
        self.cube_25_rot = self.rigid_body_states[:, 26 * 2 + 25, 3:7]
        self.cube_26_pos = self.rigid_body_states[:, 26 * 2 + 26, 0:3]
        self.cube_26_rot = self.rigid_body_states[:, 26 * 2 + 26, 3:7]
        self.cube_27_pos = self.rigid_body_states[:, 26 * 2 + 27, 0:3]
        self.cube_27_rot = self.rigid_body_states[:, 26 * 2 + 27, 3:7]

======================================================================================================================
DOF 0: pX
            DOF 1: nX
            DOF 2: pY
            DOF 3: nY
            DOF 4: pZ
            DOF 5: nZ
            DOF 6: pX_pY_0
            DOF 7: pX_pY_1
            DOF 8: pX_pY_2
            DOF 9: nY_pX_0
            DOF 10: nY_pX_1
            DOF 11: nY_pX_2
            DOF 12: pX_pZ_0
            DOF 13: pX_pZ_1
            DOF 14: pX_pZ_2
            DOF 15: nZ_pX_0
            DOF 16: nZ_pX_1
            DOF 17: nZ_pX_2
            DOF 18: nX_pY_0
            DOF 19: nX_pY_1
            DOF 20: nX_pY_2
            DOF 21: nX_nY_0
            DOF 22: nX_nY_1
            DOF 23: nX_nY_2
            DOF 24: nX_pZ_0
            DOF 25: nX_pZ_1
            DOF 26: nX_pZ_2
            DOF 27: nX_nZ_0
            DOF 28: nX_nZ_1
            DOF 29: nX_nZ_2
            DOF 30: pY_pZ_0
            DOF 31: pY_pZ_1
            DOF 32: pY_pZ_2
            DOF 33: nZ_pY_0
            DOF 34: nZ_pY_1
            DOF 35: nZ_pY_2
            DOF 36: nY_pZ_0
            DOF 37: nY_pZ_1
            DOF 38: nY_pZ_2
            DOF 39: nY_nZ_0
            DOF 40: nY_nZ_1
            DOF 41: nY_nZ_2
            DOF 42: pX_pY_pZ_0
            DOF 43: pX_pY_pZ_1
            DOF 44: pX_pY_pZ_2
            DOF 45: nZ_pX_pY_0
            DOF 46: nZ_pX_pY_1
            DOF 47: nZ_pX_pY_2
            DOF 48: nY_pX_pZ_0
            DOF 49: nY_pX_pZ_1
            DOF 50: nY_pX_pZ_2
            DOF 51: nY_nZ_pX_0
            DOF 52: nY_nZ_pX_1
            DOF 53: nY_nZ_pX_2
            DOF 54: nX_pY_pZ_0
            DOF 55: nX_pY_pZ_1
            DOF 56: nX_pY_pZ_2
            DOF 57: nX_nZ_pY_0
            DOF 58: nX_nZ_pY_1
            DOF 59: nX_nZ_pY_2
            DOF 60: nX_nY_pZ_0
            DOF 61: nX_nY_pZ_1
            DOF 62: nX_nY_pZ_2
            DOF 63: nX_nY_nZ_0
            DOF 64: nX_nY_nZ_1
            DOF 65: nX_nY_nZ_2
===============================================================================================================================
# OLD Reset
# randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        
        if self.object_type == "pen":
            rand_angle_y = torch.tensor(0.3)
            new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
                                                    self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids], self.z_unit_tensor[env_ids])
        

        # self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.root_state_tensor),
        #                                              gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5+self.num_shadow_hand_dofs]

        pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta

        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_another_dof_pos[env_ids, :] = pos
        self.object_dof_pos[env_ids, :] = to_torch([-0.59], device=self.device)
        self.object_dof_vel[env_ids, :] = to_torch([0], device=self.device)

        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]   

        self.shadow_hand_another_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]

        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        self.prev_targets[env_ids, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs*2] = pos
        self.cur_targets[env_ids, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs*2] = pos
        self.prev_targets[env_ids, self.num_shadow_hand_dofs*2:self.num_shadow_hand_dofs*2 + 2] = to_torch([-0.59], device=self.device)
        self.cur_targets[env_ids, self.num_shadow_hand_dofs*2:self.num_shadow_hand_dofs*2 + 2] = to_torch([-0.59], device=self.device)

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        another_hand_indices = self.another_hand_indices[env_ids].to(torch.int32)
        print("AATHIRA self.object_indices[env_ids] 1:", env_ids)
        all_hand_indices = torch.unique(torch.cat([hand_indices,
                                                 another_hand_indices,
                                                 self.object_indices[env_ids]]).to(torch.int32))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))  

        print("AATHIRA self.object_indices[env_ids] :", env_ids)
        all_indices = torch.unique(torch.cat([all_hand_indices,
                                              self.object_indices[env_ids],
                                              self.table_indices[env_ids]]).to(torch.int32))

        self.hand_positions[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 0:3]
        self.hand_orientations[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 3:7]
        self.hand_linvels[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 7:10]
        self.hand_angvels[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 10:13]

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))
                                              
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

==============================================================================================================
@torch.jit.script
def depth_image_to_point_cloud_GPU(camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, depth_bar:float, device:torch.device):
    # time1 = time.time()
    depth_buffer = camera_tensor.to(device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv = camera_view_matrix_inv

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    
    proj = camera_proj_matrix
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position@vinv

    points = position[:, 0:3]

    return points


=======================================================================================================
OLD compute_hand_reward()
# Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    # goal_dist = target_pos[:, 2] - object_pos[:, 2]

    right_hand_dist = torch.norm(scissors_right_handle_pos - right_hand_pos, p=2, dim=-1)
    left_hand_dist = torch.norm(scissors_left_handle_pos - left_hand_pos, p=2, dim=-1)

    right_hand_finger_dist = (torch.norm(scissors_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(scissors_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(scissors_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(scissors_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                            + torch.norm(scissors_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
    left_hand_finger_dist = (torch.norm(scissors_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(scissors_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(scissors_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(scissors_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                            + torch.norm(scissors_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
    # Orientation alignment for the cube in hand and goal cube
    # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    right_hand_dist_rew = right_hand_finger_dist
    left_hand_dist_rew = left_hand_finger_dist

    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
    up_rew = torch.zeros_like(right_hand_dist_rew)
    # Aathira : up_rew = torch.where(right_hand_finger_dist < 0.7,
               #     torch.where(left_hand_finger_dist < 0.7,
               #         (0.59 + object_dof_pos[:, 0]) * 5, up_rew), up_rew)

    # up_rew =  torch.where(right_hand_finger_dist <= 0.3, torch.norm(bottle_cap_up - bottle_pos, p=2, dim=-1) * 30, up_rew)

    # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))
    reward = 2 + up_rew - right_hand_dist_rew - left_hand_dist_rew

    resets = torch.where(up_rew < -0.5, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(right_hand_finger_dist >= 1.75, torch.ones_like(resets), resets)
    resets = torch.where(left_hand_finger_dist >= 1.75, torch.ones_like(resets), resets)
    # Find out which envs hit the goal and update successes count
    #print("AATHIRA resets shape :", resets.shape)
    #print("AATHIRA reset buffer length : ", len(reset_buf))
    resets = torch.tensor([0]*len(reset_buf)) #torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    #print("AATHIRA successes : ", successes)
    # Aathira : successes = torch.where(successes == 0, 
    #                torch.where(object_dof_pos[:, 0] > -0.3, torch.ones_like(successes), successes), successes)

    goal_resets = torch.tensor([0]*len(reset_buf)) # Aathira : torch.zeros_like(resets)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.tensor(0) # Aathira : torch.where(resets > 0, successes * resets, consecutive_successes).mean()
=======================================================================================================

def rand_row(self, tensor, dim_needed):  
        row_total = tensor.shape[0]
        return tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)),:]

    def sample_points(self, points, sample_num=1000, sample_mathed='furthest'):
        eff_points = points[points[:, 2]>0.04]
        if eff_points.shape[0] < sample_num :
            eff_points = points
        if sample_mathed == 'random':
            sampled_points = self.rand_row(eff_points, sample_num)
        elif sample_mathed == 'furthest':
            sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points.reshape(1, *eff_points.shape), sample_num)
            sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
        return sampled_points
    

    def camera_visulization(self, is_depth_image=False):
        if is_depth_image:
            camera_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_DEPTH)
            torch_depth_tensor = gymtorch.wrap_tensor(camera_depth_tensor)
            torch_depth_tensor = torch.clamp(torch_depth_tensor, -1, 1)
            torch_depth_tensor = scale(torch_depth_tensor, to_torch([0], dtype=torch.float, device=self.device),
                                                         to_torch([256], dtype=torch.float, device=self.device))
            camera_image = torch_depth_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)
        
        else:
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_COLOR)
            torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
            camera_image = torch_rgba_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)
        
        return camera_image

==========================================================================================================

def compute_point_cloud_observation(self, collect_demonstration=False):
        
        Compute the observations of all environment. The observation is composed of three parts: 
        the state values of the left and right hands, and the information of objects and target. 
        The state values of the left and right hands were the same for each task, including hand 
        joint and finger positions, velocity, and force information. The detail 428-dimensional 
        observational space as shown in below:

        Index       Description
        0 - 23	    right shadow hand dof position
        24 - 47	    right shadow hand dof velocity
        48 - 71	    right shadow hand dof force
        72 - 136	right shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
        137 - 166	right shadow hand fingertip force, torque (5 x 6)
        167 - 169	right shadow hand base position
        170 - 172	right shadow hand base rotation
        173 - 198	right shadow hand actions
        199 - 222	left shadow hand dof position
        223 - 246	left shadow hand dof velocity
        247 - 270	left shadow hand dof force
        271 - 335	left shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
        336 - 365	left shadow hand fingertip force, torque (5 x 6)
        366 - 368	left shadow hand base position
        369 - 371	left shadow hand base rotation
        372 - 397	left shadow hand actions
        398 - 404	object pose
        405 - 407	object linear velocity
        408 - 410	object angle velocity
        411 - 417	goal pose
        418 - 421	goal rot - object rot
        422 - 424	scissors right handle position
        425 - 427	scissors left handle position
        
        num_ft_states = 13 * int(self.num_fingertips / 2)  # 65
        num_ft_force_torques = 6 * int(self.num_fingertips / 2)  # 30

        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                            self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]

        fingertip_obs_start = 72  # 168 = 157 + 11
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
                    num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]
        
        hand_pose_start = fingertip_obs_start + 95
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.right_hand_pos
        self.obs_buf[:, hand_pose_start+3:hand_pose_start+4] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+4:hand_pose_start+5] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+5:hand_pose_start+6] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        self.obs_buf[:, action_obs_start:action_obs_start + 26] = self.actions[:, :26]

        # another_hand
        another_hand_start = action_obs_start + 26
        self.obs_buf[:, another_hand_start:self.num_shadow_hand_dofs + another_hand_start] = unscale(self.shadow_hand_another_dof_pos,
                                                            self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, self.num_shadow_hand_dofs + another_hand_start:2*self.num_shadow_hand_dofs + another_hand_start] = self.vel_obs_scale * self.shadow_hand_another_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs + another_hand_start:3*self.num_shadow_hand_dofs + another_hand_start] = self.force_torque_obs_scale * self.dof_force_tensor[:, 24:48]

        fingertip_another_obs_start = another_hand_start + 72
        self.obs_buf[:, fingertip_another_obs_start:fingertip_another_obs_start + num_ft_states] = self.fingertip_another_state.reshape(self.num_envs, num_ft_states)
        self.obs_buf[:, fingertip_another_obs_start + num_ft_states:fingertip_another_obs_start + num_ft_states +
                    num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, 30:]

        hand_another_pose_start = fingertip_another_obs_start + 95
        self.obs_buf[:, hand_another_pose_start:hand_another_pose_start + 3] = self.left_hand_pos
        self.obs_buf[:, hand_another_pose_start+3:hand_another_pose_start+4] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_another_pose_start+4:hand_another_pose_start+5] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_another_pose_start+5:hand_another_pose_start+6] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[2].unsqueeze(-1)

        action_another_obs_start = hand_another_pose_start + 6
        self.obs_buf[:, action_another_obs_start:action_another_obs_start + 26] = self.actions[:, 26:]

        obj_obs_start = action_another_obs_start + 26  # 144
        self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel
        self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 16] = self.scissors_right_handle_pos
        self.obs_buf[:, obj_obs_start + 16:obj_obs_start + 19] = self.scissors_left_handle_pos
        # goal_obs_start = obj_obs_start + 13  # 157 = 144 + 13
        # self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
        # self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
        point_clouds = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)
        
        if self.camera_debug:
            import matplotlib.pyplot as plt
            self.camera_rgba_debug_fig = plt.figure("CAMERA_RGBD_DEBUG")
            camera_rgba_image = self.camera_visulization(is_depth_image=False)
            plt.imshow(camera_rgba_image)
            plt.pause(1e-9)

        for i in range(self.num_envs):
            # Here is an example. In practice, it's better not to convert tensor from GPU to CPU
            points = depth_image_to_point_cloud_GPU(self.camera_tensors[i], self.camera_view_matrixs[i], self.camera_proj_matrixs[i], self.camera_u2, self.camera_v2, self.camera_props.width, self.camera_props.height, 10, self.device)
            
            if points.shape[0] > 0:
                selected_points = self.sample_points(points, sample_num=self.pointCloudDownsampleNum, sample_mathed='random')
            else:
                selected_points = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)
            
            point_clouds[i] = selected_points

        if self.pointCloudVisualizer != None :
            import open3d as o3d
            points = point_clouds[0, :, :3].cpu().numpy()
            # colors = plt.get_cmap()(point_clouds[0, :, 3].cpu().numpy())
            self.o3d_pc.points = o3d.utility.Vector3dVector(points)
            # self.o3d_pc.colors = o3d.utility.Vector3dVector(colors[..., :3])

            if self.pointCloudVisualizerInitialized == False :
                self.pointCloudVisualizer.add_geometry(self.o3d_pc)
                self.pointCloudVisualizerInitialized = True
            else :
                self.pointCloudVisualizer.update(self.o3d_pc)

        self.gym.end_access_image_tensors(self.sim)
        point_clouds -= self.env_origin.view(self.num_envs, 1, 3)

        point_clouds_start = obj_obs_start + 19
        self.obs_buf[:, point_clouds_start:].copy_(point_clouds.view(self.num_envs, self.pointCloudDownsampleNum * 3))

==============================================================================================================================================
"""