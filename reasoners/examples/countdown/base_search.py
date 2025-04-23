from typing import Literal
import os, sys, time
import json
import fire
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from lm.vllm_model import VLLM
from lm.utils_gpu import set_cuda_visible_devices
from base import BatchedReasoner
from algorithm.beam_search import BatchedBeamSearch
from algorithm.mcts import BatchedMCTS
from examples.countdown.world_model import CountdownWorldModel
from examples.countdown.search_config import BatchedCountdownConfig
from examples.countdown.utils import verify


def batch_search_countdown(
    log_dir: str,
    split: Literal["val", "test"],
    start_index: int,
    end_index: int,
    batch_size: int,
    base_model,
    search_algo_name: str,
    output_name: str,
    temperature=0.7,
    top_p=0.8,
    depth_limit: int = 4,
    n_beam: int = 5,  # used in beam search
    w_exp: float = 1.0,  # used in MCTS
    n_eval: int = 3,
    **search_algo_params,
):
    assert search_algo_name in ["bfs", "mcts"]  # ToT or RAP
    search_algo = BatchedBeamSearch if search_algo_name == "bfs" else BatchedMCTS

    num_cases = 1000
    full_dataset = json.load(
        open(
            f"{os.environ['cd_root']}/problems/{split}_cd4_t100_n{num_cases}_problems.json"
        )
    )

    assert 0 <= start_index < end_index <= len(full_dataset)
    dataset = full_dataset[start_index:end_index]
    if batch_size == -1:
        batch_size = len(dataset)

    world_model = CountdownWorldModel()
    config = BatchedCountdownConfig(
        base_model=base_model,
        temperature=temperature,
        top_p=top_p,
        depth_limit=depth_limit,
        n_eval=n_eval,
    )
    search_algo_params |= {
        "beam_size": n_beam,
        "w_exp": w_exp,
        "max_depth": depth_limit,
        "depth_limit": depth_limit,
    }
    print(search_algo_params)
    search_algo = search_algo(**search_algo_params)
    reasoner = BatchedReasoner(
        world_model=world_model, search_config=config, search_algo=search_algo
    )

    avg_counts = Counter()
    start_time = time.time()
    df = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        examples = dataset[i : i + batch_size]
        inputs = []
        for example in examples:
            inputs.append(
                [" ".join(str(num) for num in example["nums"]), str(example["target"])]
            )

        all_complete_paths, all_partial_paths = reasoner(inputs)

        for k, (input, complete_paths, partial_paths) in enumerate(
            zip(inputs, all_complete_paths, all_partial_paths)
        ):
            results = []
            for path in complete_paths:
                result = verify(path)
                df.append(
                    {
                        "case_id": tuple(path[:2]),
                        "success": result,
                        "structured": True,
                        "text": tuple(path),
                    }
                )
                results.append(result)

            for path in partial_paths:
                if len(path) == depth_limit + 2:
                    # this ensures the path is wrong
                    df.append(
                        {
                            "case_id": tuple(path[:2]),
                            "success": False,
                            "structured": False,
                            "text": tuple(path),
                        }
                    )

            # top1 path (by LLM reward) is correct
            avg_counts["top1"] += int(results[0]) if results else 0
            avg_counts["any"] += int(any(results))  # any path is correct

            log_avg = {
                key: round(value / (i + k + 1), 3) for key, value in avg_counts.items()
            }

            log_str = (
                f"Case #{start_index + i + k + 1}: {input}, {results};"
                f" running-avg {dict(log_avg)}"
                f" {(time.time() - start_time)/60:.1f}min"
            )
            with open(
                os.path.join(log_dir, f"progress_cd4_{split}_{output_name}.log"), "a"
            ) as f:
                print(log_str, file=f)

        tqdm.write(log_str)

    # directly save to csv
    df = pd.DataFrame(df)
    print(f"{len(df) =}")

    df = df.drop_duplicates()
    print(f"after dedup: {len(df) =}")

    saved_fn = os.path.join(log_dir, f"all_cd4_{split}_{output_name}.csv")
    df.to_csv(saved_fn, index=False)
    print(f"Saved {saved_fn}")

    return df


def main(
    model_pth: str,
    batch_size: int,
    split: str,
    start_index: int,
    end_index: int,
    search_algo: str,
    num_gpus: int = 1,
    swap_space: int = 16,
    dtype: str = "bfloat16",
    **kwargs,
):
    set_cuda_visible_devices(num_gpus)
    assert search_algo in ["bfs", "mcts"]

    log_dir = f"{os.environ['cd_root']}/{model_pth.split('/')[-1]}/"
    log_dir += f"{start_index}to{end_index}_{search_algo}/"
    log_dir += f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'

    os.makedirs(log_dir)
    print(sys.argv)
    with open(os.path.join(log_dir, "args.txt"), "w") as f:
        for arg in sys.argv:
            f.write(arg + "\n")

    base_model = VLLM(
        model_pth,
        num_gpus,
        swap_space=swap_space,
        dtype=dtype,
        max_length=2048,
        max_new_tokens=256,
    )

    batch_search_countdown(
        log_dir=log_dir,
        split=split,
        start_index=start_index,
        end_index=end_index,
        batch_size=batch_size,
        base_model=base_model,
        search_algo_name=search_algo,
        output_name=search_algo,
        **kwargs,
    )


if __name__ == "__main__":
    fire.Fire(main)
