import gym
import numpy as np
import pymp
import sapien.core as sapien
from transforms3d.euler import euler2quat
import trimesh

from mani_skill2.envs.pick_and_place.stack_cube import StackCubeEnv
from mani_skill2.utils.trimesh_utils import get_actor_mesh
from mani_skill2.utils.common import normalize_vector

from matplotlib import pyplot as plt


ASSET_PATH = '/home/zjia/Research/inter_seq/ManiSkill2-CoTPC/mani_skill2/assets'

def main():
    env: StackCubeEnv = gym.make(
        "StackCube-v0", obs_mode="none", control_mode="pd_joint_pos", robot='xarm7',  # xarm7
    )
    solve(env, seed=87, debug=True, vis=False)
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

    
def solve(env: StackCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"], env.control_mode
    if debug:
        pymp.logger.setLevel("DEBUG")

    # -------------------------------------------------------------------------- #
    # Utilities
    # -------------------------------------------------------------------------- #
    def render_wait(idx):
        if not vis or not debug:
            print('wait...')
            img = env.render(mode="cameras")
            plt.imsave(
                f'/home/zjia/Research/inter_seq/ManiSkill2-CoTPC/mp_demos/{idx}.png')
            return
        print("Press [c] to continue")
        viewer = env.render("human")
        while True:
            if viewer.window.key_down("c"):
                break
            env.render("human")

    done, info = False, {}

    def execute_plan(plan, gripper_action):
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
            _, _, done, info = env.step(action)
            if vis:
                env.render()

    def execute_plan2(plan, gripper_action, t):
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
        for _ in range(t):
            _, _, done, info = env.step(action)
            if vis:
                env.render()

    # -------------------------------------------------------------------------- #
    # Planner
    # -------------------------------------------------------------------------- #
    joint_names = [joint.get_name() for joint in env.agent.robot.get_active_joints()]
    # print(joint_names)
    # exit()
    planner = pymp.Planner(
        urdf=f"{ASSET_PATH}/descriptions/xarm7_with_gripper.urdf", # xarm7_with_gripper
        user_joint_names=joint_names,
        srdf=f"{ASSET_PATH}/descriptions/xarm7_with_gripper.srdf",
        ee_link_name="link_tcp", # link_tcp
        base_pose=env.agent.robot.pose,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
        timestep=env.control_timestep,
        use_convex=False,
    )
    # exit()

    OPEN_GRIPPER_POS = 1
    CLOSE_GRIPPER_POS = -1  ###
    FINGER_LENGTH = 0.025

    # -------------------------------------------------------------------------- #
    # Collision
    # -------------------------------------------------------------------------- #
    # Add collision obstacles
    # planner.scene.addBox(env.box_half_size * 2, env.cubeA.pose, name="cubeA")
    planner.scene.addBox(env.box_half_size * 2, env.cubeB.pose, name="cubeB")
    planner.scene.addBox([1, 1, 0.01], [0, 0, -0.01], name="ground")
    render_wait(1)

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

    # Search a valid pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    print(angles)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        ik_results = planner.compute_CLIK(
            grasp_pose2, env.agent.robot.get_qpos(), 1, seed=seed
        )
        if len(ik_results) == 0:
            continue
        # Avoid joint limits issue (hardcode for panda)
        # if np.abs(ik_results[0, -3]) > 2:
            # continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")

    render_wait()

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    plan = planner.plan_screw(reach_pose, env.agent.robot.get_qpos())
    execute_plan(plan, OPEN_GRIPPER_POS)
    render_wait()

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.scene.disableCollision("cubeB")
    plan = planner.plan_screw(grasp_pose, env.agent.robot.get_qpos())
    execute_plan(plan, OPEN_GRIPPER_POS)
    render_wait()

    # Close gripper
    execute_plan2(plan, CLOSE_GRIPPER_POS, 10)
    render_wait()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    plan = planner.plan_screw(lift_pose, env.agent.robot.get_qpos())
    execute_plan(plan, CLOSE_GRIPPER_POS)
    render_wait()

    # -------------------------------------------------------------------------- #
    # Stack
    # -------------------------------------------------------------------------- #
    goal_pos = env.cubeB.pose.p + [0, 0, env.box_half_size[2] * 2]
    offset = goal_pos - env.cubeA.pose.p
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    plan = planner.plan_screw(align_pose, env.agent.robot.get_qpos())
    execute_plan(plan, CLOSE_GRIPPER_POS)
    render_wait()

    # release
    execute_plan2(plan, OPEN_GRIPPER_POS, 10)
    render_wait()

    print(info)
    return info


if __name__ == "__main__":
    main()