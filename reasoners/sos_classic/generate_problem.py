"""
Step 1: Script for generating the problem set for the countdown task.
"""

import json
import argparse
import random
import os

import tqdm
from collections import defaultdict, Counter

from countdown import CountDown
from countdown_utils import *


parser = argparse.ArgumentParser()

# data args
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument(
    "--data_dir", type=str, default=None, help="Directory to store data"
)

# countdown specific
parser.add_argument("--cd_inputs", type=int, default=4, help="number of inputs")
parser.add_argument("--max_target", type=int, default=100, help="Maximum target number")
parser.add_argument(
    "--num_samples", type=int, default=1000, help="Number of data samples to generate"
)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    # set random seed
    random.seed(args.seed)
    target_nums = [i for i in range(10, args.max_target + 1)]
    if 24 in target_nums:
        target_nums.remove(24)  # Remove 24 to handle it separately

    # save 10% of target numbers for validation
    random.shuffle(target_nums)
    val_target_nums = target_nums[: len(target_nums) // 10]
    train_nums = target_nums[len(target_nums) // 10 :]
    # Add 24 explicitly to the validation
    val_target_nums.append(24)

    splits = ["train", "test", "val"]
    target_list = [train_nums, val_target_nums, train_nums]
    print(target_list)

    data_samples = {}
    problems = defaultdict(list)

    for split, target_nums in zip(splits, target_list):

        data_samples[split] = []
        if split == "train":
            num_samples = args.num_samples
        else:
            num_samples = 1000

        print(f"start generating {split} split for {num_samples} problems")
        for _ in tqdm.tqdm(range(num_samples)):

            cd = CountDown(args.max_target, args.cd_inputs)
            target = random.choice(target_nums)
            nums, solution = cd.generate(target)

            if split == "val":
                while any(
                    Counter(nums) == Counter(s["nums"]) and target == s["target"]
                    for s in problems["train"]
                ):  # assure the validation problem is different from training problem
                    target = random.choice(target_nums)
                    nums, solution = cd.generate(target)

            problems[split].append(
                {
                    "nums": nums,
                    "target": target,
                    "solution": solution,
                    "optimal_path": cd.convert_to_path(target, nums, solution),
                }
            )

        os.makedirs(args.data_dir, exist_ok=True)
        with open(
            f"{args.data_dir}/problems/{split}_cd{args.cd_inputs}_t{args.max_target}_n{num_samples}_problems.json",
            "w",
        ) as f:
            json.dump(problems[split], f, indent=4)
