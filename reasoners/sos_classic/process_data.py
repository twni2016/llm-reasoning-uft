import re
import os
import json
import argparse
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

parser = argparse.ArgumentParser()

parser.add_argument("--cd_inputs", type=int, default=4, help="")
parser.add_argument("--search", type=str, default="random", help="[dfs, bfs]")
parser.add_argument("--split", type=str, default="val", help="[train, val, test]")
parser.add_argument(
    "--downsample_size",
    type=int,
    default=3,
    help="how many negative examples to keep per case.",
)


def extract_path(search_path):
    search_steps = search_path.split("\n")

    index = [i for i, s in enumerate(search_steps) if "Goal Reached" in s]
    if len(index) == 0:
        succ_path = []
    else:
        assert len(index) == 1
        index = index[0]
        final_step = (
            search_steps[index - 1].split("Exploring Operation: ")[-1].split(",")[0]
        )
        for i in reversed(range(index)):
            if "Current State:" in search_steps[i]:
                succ_path = eval(search_steps[i].split("Operations: ")[-1])
                succ_path.append(final_step)
                break

    failed_paths = []
    # find the last steps
    indices = [i for i, s in enumerate(search_steps) if "No Solution" in s]
    for index in indices:
        final_step = (
            search_steps[index - 1].split("Exploring Operation: ")[-1].split(",")[0]
        )
        for i in reversed(range(index)):
            if "Current State:" in search_steps[i]:
                failed_path = eval(search_steps[i].split("Operations: ")[-1])
                failed_path.append(final_step)
                failed_paths.append(failed_path)
                break

    return succ_path, failed_paths


def postprocess(nums: list, target: int, path: list):
    ## first append left numbers to each step
    path = append_goal(nums, path)

    ## then add final equation (we don't need input numbers and target)
    answer = parse_game_input(path)
    test_answer(nums, answer)

    ## finally add spaces
    path = [add_spaces_to_arithmetic(s) for s in path]

    return [" ".join([str(num) for num in nums]), str(target), *path, answer]


def append_goal(nums, steps):
    def split_expression(expression):
        operators = "+-/*"
        for op in operators:
            if op in expression:
                numbers = expression.split(op)
                return int(numbers[0]), op, int(numbers[1])

        # If no operator is found, raise an exception
        raise ValueError("Invalid expression: No operator found")

    new_steps = []
    nums = deepcopy(nums)
    for s in steps:
        lhs, rhs = s.split("=")
        rhs = int(rhs)
        num1, _, num2 = split_expression(lhs)

        nums.remove(num1)  # remove the first appearance
        nums.remove(num2)
        nums.append(rhs)  # push to the stack

        s += " (left: " + " ".join([str(i) for i in nums]) + ")"
        new_steps.append(s)
    return new_steps


def parse_game_input(path):
    # this is a greedy algorithm to output one possible answer

    # Extract operations and results
    operations = re.findall(r"(\d+)\s*([*/+-])\s*(\d+)\s*=\s*(\d+)", "\n".join(path))

    # Construct the operation string
    operation_str = ""
    prev_results = defaultdict(
        list
    )  # Dictionary to store previous results and their expressions

    for i, (num1, op, num2, result) in enumerate(operations):
        if i == 0:
            current_op = f"({num1} {op} {num2})"
            prev_results[result].append(current_op)
            operation_str = current_op
        else:
            # Replace any number if it matches one of previous results
            if num1 in prev_results and len(prev_results[num1]) > 0:
                num1_expr = prev_results[num1].pop()
            else:
                num1_expr = num1

            if num2 in prev_results and len(prev_results[num2]) > 0:
                num2_expr = prev_results[num2].pop()
            else:
                num2_expr = num2

            current_op = f"({num1_expr} {op} {num2_expr})"

            prev_results[result].append(current_op)
            operation_str = current_op

    # remove final parentheses
    operation_str = operation_str[1:-1]
    answer = f"{operation_str} = {result}"

    return answer


def test_answer(nums: list, answer: str):
    # first check lhs = rhs (the target number is checked already)
    lhs, rhs = answer.split(" = ")
    assert int(eval(lhs)) == int(rhs), answer

    # then check if nums are all used once
    used_nums = re.findall(r"\d+", lhs)
    input_nums = [str(num) for num in nums]
    assert Counter(used_nums) == Counter(input_nums), answer


def add_spaces_to_arithmetic(s):
    # Use regex to find arithmetic expressions (number operator number = number)
    pattern = r"(\d+)([*/+-])(\d+)(=)(\d+)"

    def add_spaces(match):
        num1, op, num2, eq, result = match.groups()
        return f"{num1} {op} {num2} {eq} {result}"

    return re.sub(pattern, add_spaces, s)


def convert_sos_to_sft_format(
    cd_inputs: int, search: str, split: str, downsample_size: int
):
    if split == "train":
        num_samples = 500_000
    else:
        num_samples = 1000
    fn = f"../../data/countdown/inf_logs/{search}/{split}_cd{cd_inputs}_t100_n{num_samples}_{search}.json"
    print("converting", fn)

    data = json.load(open(fn))
    success_trajs, failed_trajs = [], []

    for d in tqdm(data):
        succ_path, failed_paths = extract_path(d["search_path"])

        if len(succ_path) == cd_inputs - 1:
            # sanity check
            assert int(succ_path[-1].split("=")[-1]) == d["target"]

            traj = postprocess(d["nums"], d["target"], succ_path)
            success_trajs.append(traj)

        for f_t in failed_paths:
            if len(f_t) != cd_inputs - 1:
                continue
            # sanity check (SoS assume integers)
            assert int(f_t[-1].split("=")[-1]) != d["target"]

            traj = postprocess(d["nums"], d["target"], f_t)
            failed_trajs.append(traj)

    succ_df = pd.DataFrame(
        [
            {
                "case_id": tuple(traj[:2]),
                "success": True,
                "structured": True,
                "text": tuple(traj),
            }
            for traj in success_trajs
        ]
    )
    fail_df = pd.DataFrame(
        [
            {
                "case_id": tuple(traj[:2]),
                "success": False,
                "structured": True,
                "text": tuple(traj),
            }
            for traj in failed_trajs
        ]
    )
    print(f"{len(succ_df)=}, {len(fail_df)=}")

    succ_df = succ_df.drop_duplicates()
    fail_df = fail_df.drop_duplicates()
    print(f"after dedup: {len(succ_df)=}, {len(fail_df)=}")

    # dedup
    saved_fn = f"../../data/countdown/corpora/{search}/correct_{split}.csv"
    os.makedirs(os.path.dirname(saved_fn), exist_ok=True)
    succ_df.to_csv(saved_fn, index=False)
    print(f"Saved {saved_fn}")

    # downsample by case
    percentiles = np.arange(0.1, 1.0, 0.1)

    fail_df_gb = fail_df.groupby("case_id", group_keys=False)
    print(fail_df_gb.size().describe(percentiles).round())

    fail_df = fail_df_gb.apply(
        lambda x: x.sample(min(len(x), downsample_size), random_state=42),
    )
    fail_df_gb = fail_df.groupby("case_id", group_keys=False)
    print(f"\nDownsample failed data to {len(fail_df)=}")
    print(fail_df_gb.size().describe(percentiles).round())

    saved_fn = f"../../data/countdown/corpora/{search}/failed_{split}.csv"
    fail_df.to_csv(saved_fn, index=False)
    print(f"Saved {saved_fn}")


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    convert_sos_to_sft_format(
        cd_inputs=args.cd_inputs,
        search=args.search,
        split=args.split,
        downsample_size=args.downsample_size,
    )
