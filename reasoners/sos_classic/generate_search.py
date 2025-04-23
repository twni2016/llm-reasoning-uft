"""
Script for generating BFS / DFS data for the countdown task given the problem set.
"""

import json
import argparse
import random
import os, time
import tqdm
import numpy as np
from countdown_utils import *
from countdown_bfs import bfs
from countdown_dfs import dfs


parser = argparse.ArgumentParser()

# data args
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument(
    "--save_dir", type=str, default=None, help="Directory to store data"
)
parser.add_argument(
    "--data_dir", type=str, default=None, help="Directory to load problem data"
)

# countdown specific
parser.add_argument("--cd_inputs", type=int, default=4, help="number of inputs")
parser.add_argument("--max_target", type=int, default=100, help="Maximum target number")
parser.add_argument(
    "--num_samples", type=int, default=1000, help="Number of data samples to generate"
)

# search args
parser.add_argument(
    "--search", type=str, default="random", help="Search type [dfs, bfs]"
)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    random.seed(args.seed)

    if args.cd_inputs == 2:
        # naive calculation of max nodes: 2c2 x 4 = 4
        max_rating = 4
    elif args.cd_inputs == 3:
        # naive calculation of max nodes: 3c2 x 4 x 4 = 48
        max_rating = 48
    elif args.cd_inputs == 4:
        # naive calculation of max nodes: 4c2 x 4 x 3c2 x 4 x 4 = 1152
        max_rating = 1152
    elif args.cd_inputs == 5:
        # naive calculation of max nodes: 5c2 x 4 x 4c2 x 4 x 3c2 x 4 x 4 = 46080
        max_rating = 46080

    data_samples = {}
    problems = {}
    average_succ = {}
    average_reward = {}

    for split in ["train", "test", "val"]:
        if split == "train":
            num_samples = args.num_samples
        else:
            num_samples = 1000

        problems[split] = json.load(
            open(
                f"{args.data_dir}/problems/{split}_cd{args.cd_inputs}_t{args.max_target}_n{num_samples}_problems.json"
            )
        )
        data_samples[split] = []
        average_succ[split] = []
        average_reward[split] = []

        print(f"start generating {split} split for {num_samples} problems")
        start_time = time.time()

        for t in tqdm.tqdm(range(num_samples)):
            problem = problems[split][t]
            target, nums = problem["target"], problem["nums"]

            if args.search == "dfs":
                search_path = dfs(
                    target, nums, heuristic=sum_heuristic, threshold=target
                )
                search = dfs
                heuristic = sum_heuristic
            elif args.search == "bfs":
                search_path = bfs(target, nums, 5, heuristic=mult_heuristic)
                search = bfs
                heuristic = mult_heuristic
                beam_size = 5
            elif args.search == "random":
                heuristic = random.choice([sum_heuristic, mult_heuristic])
                search = random.choice([dfs, bfs])
                if search == dfs:
                    search_path = dfs(
                        target, nums, heuristic=heuristic, threshold=target
                    )
                elif search == bfs:
                    beam_size = random.choice([1, 2, 3, 4, 5])
                    search_path = bfs(target, nums, beam_size, heuristic=heuristic)
            else:
                raise ValueError(f"Search type {args.search} not supported")

            if "Goal Reached" in search_path:
                rating = 1.0 - simple_rating(search_path) / max_rating
                rating = max(0, rating)
            else:
                rating = 0.0
            average_succ[split].append(int(rating > 0))
            average_reward[split].append(rating)

            search_type = search.__name__
            if search_type == "bfs":
                search_type += f"_{beam_size}"

            data_samples[split].append(
                {
                    "nums": nums,
                    "target": target,
                    "solution": problem["solution"],
                    "search_path": search_path,
                    "rating": rating,
                    "search_type": search_type,
                    "optimal_path": problem["optimal_path"],
                    "heuristic": heuristic.__name__,
                }
            )

        dirname = f"{args.data_dir}/{args.search}/"
        os.makedirs(dirname, exist_ok=True)
        filename = (
            dirname
            + f"{split}_cd{args.cd_inputs}_t{args.max_target}_n{num_samples}_{args.search}"
        )
        with open(filename + ".json", "w") as f:
            json.dump(data_samples[split], f, indent=4)

        with open(filename + ".txt", "w") as f:
            f.write(f"total examples: {len(average_succ[split])}\n")
            f.write(f"average succ: {np.mean(average_succ[split]):.4f}\n")
            f.write(f"average reward: {np.mean(average_reward[split]):.4f}\n")
            f.write(
                f"average time: {(time.time() - start_time) / len(average_succ[split]):.4f}sec\n"
            )
