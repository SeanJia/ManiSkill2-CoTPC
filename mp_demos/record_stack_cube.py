import argparse
import multiprocessing as mp
import time
from copy import deepcopy

import gym
import numpy as np

from mani_skill2.utils.wrappers import RecordEpisode

# isort: off
from stack_cube import solve


def _main(args):
    env_id = "StackCube-v1"
    env = gym.make(
        env_id, obs_mode=args.obs_mode, control_mode=args.control_mode, robot='xarm7')

    if args.record_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        record_dir = f"demos/{env_id}/{timestamp}"
    else:
        record_dir = args.record_dir

    env = RecordEpisode(
        env,
        record_dir,
        save_trajectory=(not args.no_traj),
        trajectory_name="trajectory",
        save_video=(not args.no_video),
        save_on_reset=False,
        clean_on_close=False,  # A bug in record.py; clean both h5 and json in merge_trajectory.py
        render_mode=args.render_mode,
    )

    n_success = 0
    n = 0
    seed = args.start_seed

    while n_success < args.num_episodes:
        info = solve(env, seed=seed, debug=False, vis=False)
        success = info["success"]
        timeout = "TimeLimit.truncated" in info

        # Save video
        if not args.no_video:
            elapsed_steps = info["elapsed_steps"]
            suffix = "seed={}-success={}-steps={}".format(seed, success, elapsed_steps)
            env.flush_video(suffix, verbose=True)

        # Save trajectory
        if not args.no_traj:
            save_traj = args.no_discard or (success and (not timeout))
            if save_traj:
                env.flush_trajectory(verbose=True)

        n_success += int(success and (not timeout))
        n += 1
        seed += 1

    env.close()
    print("success rate", n_success / n)
    return n_success, n


def _main_mp(args):
    env_id = "StackCube-v1"

    if args.record_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        record_dir = f"demos/{env_id}/{timestamp}"
    else:
        record_dir = args.record_dir

    proc_args = []
    for i in range(args.parallel):
        _arg = deepcopy(args)
        _arg.record_dir = record_dir + "/" + str(i)
        _arg.start_seed = i * (args.num_episodes * 2)
        proc_args.append(_arg)
    pool = mp.Pool(args.parallel)
    success_rates = pool.map(_main, proc_args)
    print(
        "success rate",
        sum(x[0] for x in success_rates) / sum(x[1] for x in success_rates),
    )
    pool.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("-c", "--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument("-r", "--record-dir", type=str)
    parser.add_argument("-n", "--num-episodes", type=int, default=1)
    parser.add_argument("--render-mode", type=str, default="cameras")
    parser.add_argument("--no-traj", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-discard", action="store_true")
    parser.add_argument("-p", "--parallel", type=int)
    parser.add_argument("-s", "--start-seed", type=int, default=0)
    args = parser.parse_args()

    if args.parallel:
        _main_mp(args)
    else:
        _main(args)


if __name__ == "__main__":
    main()

# python record_stack_cube.py --no-video --parallel=10 --num-episodes=50 \
#   --record-dir=/home/zjia/Research/inter_seq/generated_demos/stack_cube

# python trajectory/merge_trajectory.py \
#     --input-dirs=/home/zjia/Research/inter_seq/generated_demos/stack_cube \
#     --output-path=/home/zjia/Research/inter_seq/generated_demos/stack_cube/trajectory.h5 \
#     --pattern=*/*.h5

# python trajectory/replay_trajectory.py \
#     --traj-path /home/zjia/Research/inter_seq/generated_demos/stack_cube/trajectory.h5 \
# 	--save-traj --target-control-mode pd_joint_delta_pos --obs-mode state --num-procs 10