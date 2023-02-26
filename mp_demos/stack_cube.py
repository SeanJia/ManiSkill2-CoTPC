import gym
import numpy as np
import pymp
import sapien.core as sapien
from transforms3d.euler import euler2quat
import transforms3d.quaternions as Q 
import trimesh

import mani_skill2.envs
from mani_skill2.utils.trimesh_utils import get_actor_mesh
from mani_skill2.utils.common import normalize_vector


ASSET_PATH = '/home/zjia/Research/inter_seq/ManiSkill2-CoTPC/mani_skill2/assets'

def main():
    env = gym.make(
        "StackCube-v1", obs_mode="none", control_mode="pd_joint_pos", robot='xarm7')
    count = 0
    size = 50
    # info = solve(env, seed=0, debug=False, vis=False)
    # env.reset(seed=0, reconfigure=True)
    # all_actions = np.load('/home/zjia/Research/inter_seq/stackcube_mp_actions.npy')
    # for action in all_actions:
        # _, _, _, info = env.step(action)
        # print(action)
        # print(env.agent.robot.get_qpos())
        # assert np.allclose(action[:7], env.agent.robot.get_qpos()[:7], rtol=0.05, atol=0.01)
        # print()
    # print(info)
    # exit()
    for seed in range(size):
        info = solve(env, seed=seed, debug=False, vis=False)
        if info['success']: 
            print(seed, 'Good')
            count += 1
        else:
            print(seed, 'Bad')
    print(count / size)
    env.close()

def get_actor_obb(actor: sapien.Actor, to_world_frame=True, vis=False):
    mesh = get_actor_mesh(actor, to_world_frame=to_world_frame)
    assert mesh is not None, "can not get actor mesh for {}".format(actor)

    obb: trimesh.primitives.Box = mesh.bounding_box_oriented

    if vis:
        obb.visual.vertex_colors = (255, 0, 0, 10)
        trimesh.Scene([mesh, obb]).show()

    return obb

def compute_grasp_info_by_obb(
    obb: trimesh.primitives.Box,
    approaching=(0, 0, -1),
    target_closing=None,
    depth=0.0,
    ortho=True,
):
    """Compute grasp info given an oriented bounding box.
    The grasp info includes axes to define grasp frame, namely approaching, closing, orthogonal directions and center.
    Args:
        obb: oriented bounding box to grasp
        approaching: direction to approach the object
        target_closing: target closing direction, used to select one of multiple solutions
        depth: displacement from hand to tcp along the approaching vector. Usually finger length.
        ortho: whether to orthogonalize closing  w.r.t. approaching.
    """
    # NOTE(jigu): DO NOT USE `x.extents`, which is inconsistent with `x.primitive.transform`!
    extents = np.array(obb.primitive.extents)
    T = np.array(obb.primitive.transform)

    # Assume normalized
    approaching = np.array(approaching)

    # Find the axis closest to approaching vector
    angles = approaching @ T[:3, :3]  # [3]
    inds0 = np.argsort(np.abs(angles))
    ind0 = inds0[-1]

    # Find the shorter axis as closing vector
    inds1 = np.argsort(extents[inds0[0:-1]])
    ind1 = inds0[0:-1][inds1[0]]
    ind2 = inds0[0:-1][inds1[1]]

    # If sizes are close, choose the one closest to the target closing
    if target_closing is not None and 0.99 < (extents[ind1] / extents[ind2]) < 1.01:
        vec1 = T[:3, ind1]
        vec2 = T[:3, ind2]
        if np.abs(target_closing @ vec1) < np.abs(target_closing @ vec2):
            ind1 = inds0[0:-1][inds1[1]]
            ind2 = inds0[0:-1][inds1[0]]
    closing = T[:3, ind1]

    # Flip if far from target
    if target_closing is not None and target_closing @ closing < 0:
        closing = -closing

    # Reorder extents
    extents = extents[[ind0, ind1, ind2]]

    # Find the origin on the surface
    center = T[:3, 3].copy()
    half_size = extents[0] * 0.5
    center = center + approaching * (-half_size + min(depth, half_size))

    if ortho:
        closing = closing - (approaching @ closing) * approaching
        closing = normalize_vector(closing)

    grasp_info = dict(
        approaching=approaching, closing=closing, center=center, extents=extents
    )
    return grasp_info


def replay_trajectory(env, traj):
    env.set_state(traj["state"][0])
    for i in range(traj["action"].shape[0]):
        env.render()
        env.step(traj["action"][i])
    env.close()

def print_info(stage_name, env):
    print(f'{stage_name} ..............')
    print('robot qpos:', env.agent.robot.get_qpos())
    print('robot eef pos:', env.agent.robot.get_links()[-1].get_pose())
    print('cubeA pose:', env.cubeA.get_pose())
    print('cubeB pose:', env.cubeB.get_pose())
    print()

def solve(env, seed=None, debug=False, vis=False):
    # all_actions = []
    env.reset(seed=seed, reconfigure=True)
    assert env.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"], env.control_mode
    if debug:
        pymp.logger.setLevel("DEBUG")

    # -------------------------------------------------------------------------- #
    # Utilities
    # -------------------------------------------------------------------------- #
    def render_wait():
        if not vis or not debug:
            print('wait...')
            # img = env.render(mode="cameras")
            # plt.imsave(
                # f'/home/zjia/Research/inter_seq/ManiSkill2-CoTPC/mp_demos/{idx}.png')
            return
        print("Press [c] to continue")
        viewer = env.render("human")
        while True:
            if viewer.window.key_down("c"):
                break
            env.render("human")

    done, info = False, {}

    def execute_plan(plan, gripper_action, debug=False):
        """Arm and gripper action."""
        nonlocal done, info

        if plan["status"] != "success":
            print(plan["status"], plan.get("reason"))
            return
        n = len(plan["position"])
        for i in range(n):
            qpos = plan["position"][i]
            if env.control_mode == "pd_joint_pos_vel":
                if "velocity" in plan:
                    qvel = plan["velocity"][i]
                else:  # in case n == 1
                    qvel = np.zeros_like(qpos)
                action = action = np.hstack([qpos, qvel, gripper_action])
            else:
                action = np.hstack([qpos, gripper_action])
            # all_actions.append(action)
            _, _, done, info = env.step(action)
            if debug:
                print(info)
            if vis:
                env.render()

    def execute_plan2(plan, gripper_action, t, debug=False):
        """Gripper action at the last step of arm plan."""
        nonlocal done, info

        if plan is None or plan["status"] != "success":
            qpos = env.agent.robot.get_qpos()[:-2]  # hardcode
        else:
            qpos = plan["position"][-1]
        if env.control_mode == "pd_joint_pos_vel":
            action = np.hstack([qpos, np.zeros_like(qpos), gripper_action])
        else:
            action = np.hstack([qpos, gripper_action])
        for i in range(t):
            # all_actions.append(action)
            _, _, done, info = env.step(action)
            # print(env.agent.robot.get_qpos())
            if debug:
                print(info)
            if vis:
                env.render()

    # -------------------------------------------------------------------------- #
    # Planner
    # -------------------------------------------------------------------------- #
    joint_names = [joint.get_name() for joint in env.agent.robot.get_active_joints()]

    planner = pymp.Planner(
        urdf=f"{ASSET_PATH}/descriptions/xarm7_with_gripper.urdf",
        user_joint_names=joint_names,
        srdf=f"{ASSET_PATH}/descriptions/xarm7_with_gripper.srdf",
        ee_link_name="link_tcp",
        base_pose=env.agent.robot.pose,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
        timestep=env.control_timestep,
        use_convex=False,
    )
    # print([link.get_name() for link in env.agent.robot.get_links()])
    # print(env.agent.robot.get_links()[-1].get_pose())
    # exit()

    OPEN_GRIPPER_POS = -1
    CLOSE_GRIPPER_POS = 1 
    FINGER_LENGTH = 0.025

    # -------------------------------------------------------------------------- #
    # Collision
    # -------------------------------------------------------------------------- #
    # Add collision obstacles
    # planner.scene.addBox(env.box_half_size * 2, env.cubeA.pose, name="cubeA")
    planner.scene.addBox(env.box_half_size * 2, env.cubeB.pose, name="cubeB")
    planner.scene.addBox([1, 1, 0.01], [0, 0, -0.01], name="ground")

    # -------------------------------------------------------------------------- #
    # Grasp pose
    # -------------------------------------------------------------------------- #
    approaching = (0, 0, -1)
    target_closing = env.tcp.pose.to_transformation_matrix()[:3, 1]
    obb = get_actor_obb(env.cubeA)
    grasp_info = compute_grasp_info_by_obb(
        obb, approaching, target_closing, depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    q = Q.qmult(grasp_pose.q, euler2quat(0, np.pi / 2, 0))
    p = grasp_pose.p
    p[-1] = 0 # -0.02
    grasp_pose = sapien.Pose(p=p, q=q)
    # exit()

    # [-0.0763505, 0.0132541, 0.0369471], [-0.0347915, 0.687281, 0.724361, -0.0416716]
    # test = sapien.Pose(p=[-0.1, -0.1, 0.1], q=[-0.0347915, 0.687281, 0.724361, -0.0416716])#euler2quat(0, np.pi, 0)) #[-0.334312, 0.625015, 0.609417, -0.355249])
    # test = sapien.Pose(p=[0.4, 0.0, 0.1], q=euler2quat(0, np.pi, 0)) #[-0.334312, 0.625015, 0.609417, -0.355249])
    ik_results = planner.compute_CLIK(
        grasp_pose, env.agent.robot.get_qpos(), 1, seed=seed
    )
    
    assert len(ik_results) > 0
    # print(ik_results)
    # exit()

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.09])
    plan = planner.plan_screw(reach_pose, env.agent.robot.get_qpos())
    execute_plan(plan, OPEN_GRIPPER_POS)
    # print_info('reach', env)
    # exit()

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.scene.disableCollision("cubeB")
    plan = planner.plan_screw(grasp_pose, env.agent.robot.get_qpos())
    execute_plan(plan, OPEN_GRIPPER_POS, debug=False)
    # print_info('grasp', env)
    # exit()

    # Close gripper
    execute_plan2(plan, CLOSE_GRIPPER_POS, 20, debug=False)
    # print_info('close', env)
    # exit()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    plan = planner.plan_screw(lift_pose, env.agent.robot.get_qpos())
    execute_plan(plan, CLOSE_GRIPPER_POS, debug=False)
    # print_info('lift', env)
    # exit()

    # -------------------------------------------------------------------------- #
    # Stack
    # -------------------------------------------------------------------------- #
    goal_pos = env.cubeB.pose.p + [0, 0, env.box_half_size[2] * 2]
    # offset = goal_pos - env.cubeA.pose.p
    # align_pose = sapien.Pose([0, 0, 0.1]) * sapien.Pose(lift_pose.p + offset, lift_pose.q)
    offset = lift_pose.p - env.cubeA.pose.p
    # align_pose = sapien.Pose([0, 0, 0.02]) * 
    align_pose = sapien.Pose(goal_pos + offset, lift_pose.q) 
    # print(align_pose, 'align')
    # print_info('stack', env)
    plan = planner.plan_screw(align_pose, env.agent.robot.get_qpos())
    execute_plan(plan, CLOSE_GRIPPER_POS, debug=False)
    # print_info('stack', env)
    # print(align_pose, 'align')
    # exit()

    # release
    execute_plan2(plan, OPEN_GRIPPER_POS, 10)

    # all_actions = np.stack(all_actions)
    # print(all_actions.shape, '....................')
    # np.save('/home/zjia/Research/inter_seq/stackcube_mp_actions.npy', all_actions)
    return info


if __name__ == "__main__":
    main()