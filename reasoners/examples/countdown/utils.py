import re
from collections import Counter
from copy import deepcopy

eps = 1e-6


def verify(path):
    """
    path: [nums, target, steps, answer]
        nums: "num num num num" or "num num num num num"
        target: str
        steps: list of str
        answer: str
    accept float numbers (sos only allows integers)
    """
    nums, target = path[:2]
    reasoning, answer = path[2:-1], path[-1]

    # First verify reasoning steps
    ## game24 and cd4 should have 3 steps
    input_nums = nums.split(" ")
    reason_steps = len(input_nums) - 1
    if len(reasoning) != reason_steps:
        # print("1")
        return False
    left_numbers = deepcopy(input_nums)

    for i in range(0, reason_steps):
        # parse lhs = rhs (left: numbers)
        pattern = r"(.*?)\s*=\s*(.*?)\s*\(left:\s*([\d\s\.]+)\)"
        match = re.match(pattern, reasoning[i])
        if not match:
            # print("2")
            return False

        lhs = match.group(1).strip()
        # parse <num 1> <op> <num 2> in lhs
        lhs_pattern = r"(\d+\.?\d*)\s*([\+\-\*/])\s*(\d+\.?\d*)"
        lhs_match = re.match(lhs_pattern, lhs)
        if not lhs_match:
            # print("3")
            return False

        ## using left numbers?
        lhs_numbers = [lhs_match.group(1), lhs_match.group(3)]
        if bool(Counter(lhs_numbers) - Counter(left_numbers)):
            # not subseteq
            # print("4")
            return False

        ## lhs = rhs?
        rhs = match.group(2).strip()
        try:
            lhs_value = float(eval(lhs))
            if abs(lhs_value - float(rhs)) >= eps:
                # print("5")
                return False
        except:
            return False

        ## final rhs = target?
        if i == reason_steps - 1:
            if abs(float(target) - float(rhs)) >= eps:
                # print("6") # this is tabular bfs/dfs's failure
                return False

        ## left numbers are correct?
        new_left_numbers = match.group(3).strip().split(" ")
        if Counter(left_numbers) - Counter(lhs_numbers) + Counter([rhs]) != Counter(
            new_left_numbers
        ):
            # print("7")
            return False
        left_numbers = new_left_numbers

    # Then verify the answer
    ## lhs = rhs = target?
    try:
        lhs, rhs = answer.split(" = ")
        if abs(eval(lhs) - float(rhs)) >= eps:
            # print("8")
            return False
        if abs(float(target) - float(rhs)) >= eps:
            # print("9")
            return False
    except:
        return False

    ## nums are all used once?
    try:
        used_nums = re.findall(r"\d+", lhs)
        if Counter(used_nums) != Counter(input_nums):
            # print("10")
            return False
    except:
        return False

    return True


if __name__ == "__main__":
    ## passed the test on BFS/DFS training datasets
    import pandas as pd
    import ast
    from tqdm import tqdm

    for search in ["bfs", "dfs"]:
        print(search)

        correct_df = pd.read_csv(f"data/countdown/corpora/{search}/correct_train.csv")
        correct_df["text"] = correct_df["text"].apply(ast.literal_eval)

        for path in tqdm(correct_df["text"]):
            assert verify(path), path

        failed_df = pd.read_csv(f"data/countdown/corpora/{search}/failed_train.csv")
        failed_df["text"] = failed_df["text"].apply(ast.literal_eval)

        for path in tqdm(failed_df["text"]):
            assert not verify(path), path
