from lm.vllm_model import VLLM
from lm.utils_gpu import set_cuda_visible_devices
from datetime import datetime
import os, sys, time
import pandas as pd
from examples.countdown.utils import verify
from collections import Counter
from tqdm import tqdm
import json
import fire


def batch_cot_game24(
    log_dir: str,
    prompt_name: str,
    start_index: int,
    end_index: int,
    batch_size: int,
    base_model,
    output_name: str,
    n=1,
    temperature=0,
    top_p=1.0,
    dedup: bool = True,
    **kwargs,
):
    with open("examples/game24/prompts/game24.json") as f:
        prompt_pool = json.load(f)

    assert prompt_name in ["zeroshot", "fewshot"]
    if prompt_name == "zeroshot":
        prompts = prompt_pool["ft_prompt"]
    else:
        prompts = prompt_pool["cot_prompt"]

    full_dataset = list(pd.read_csv("examples/game24/metadata/24.csv")["Puzzles"])

    assert 0 <= start_index < end_index <= len(full_dataset)
    dataset = full_dataset[start_index:end_index]
    if batch_size == -1:
        batch_size = len(dataset)

    avg_counts = Counter()
    start_time = time.time()
    df = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        examples = dataset[i : i + batch_size]
        inputs = []
        for example in examples:
            input = prompts.format(input=example)
            inputs.append(input)

        responses = base_model.generate(
            inputs,
            stop=["\n\n", "\nInput", "(Note:"],
            n=n,
            temperature=temperature,
            top_p=top_p,
        )

        for k, (example, response) in enumerate(zip(examples, responses)):
            results = []

            for o in response.outputs:
                output = o.text.strip()

                if "\nAnswer:" in output:
                    for index, item in enumerate(output.split("\n")):
                        if "Answer:" in item:  # find the first appearance
                            break

                    output = "\n".join(output.split("\n")[: index + 1])
                    answer = output.split("Answer:")[1].strip("` \t\n\r")

                    path_to_verify = (
                        [example, "24"] + output.split("\n")[:-1] + [answer]
                    )
                    result = verify(path_to_verify)

                    path = [example] + output.split("\n")[:-1] + [answer]
                    df.append(
                        {
                            "case_id": start_index + i + k + 1,
                            "success": result,
                            "structured": True,
                            "text": tuple(path),
                        }
                    )
                    results.append(result)
                else:
                    path = [example] + output.split("\n")
                    df.append(
                        {
                            "case_id": start_index + i + k + 1,
                            "success": False,
                            "structured": False,
                            "text": tuple(path),
                        }
                    )
                    results.append(False)

            avg_counts["avg"] += sum(results) / len(results)
            avg_counts["any"] += int(any(results))  # any path is correct

            log_avg = {
                key: round(value / (i + k + 1), 3) for key, value in avg_counts.items()
            }
            log_str = (
                f"Case #{start_index + i + k + 1}: {example=}, {results};"
                f" running-avg {dict(log_avg)}"
                f" {(time.time() - start_time)/60:.1f}min"
            )
            with open(
                os.path.join(log_dir, f"progress_game24_{output_name}.log"), "a"
            ) as f:
                print(log_str, file=f)

        tqdm.write(log_str)

    # directly save to csv
    df = pd.DataFrame(df)
    print(f"{len(df) =}")

    if dedup:
        df = df.drop_duplicates()
        print(f"after dedup: {len(df) =}")

    saved_fn = os.path.join(log_dir, f"all_game24_{output_name}.csv")
    df.to_csv(saved_fn, index=False)
    print(f"Saved {saved_fn}")

    return df


def main(
    model_pth: str,
    batch_size: int,
    start_index: int,
    end_index: int,
    num_gpus: int = 1,
    swap_space: int = 16,
    dtype: str = "bfloat16",
    prompt_name: str = "fewshot",
    **kwargs,
):
    set_cuda_visible_devices(num_gpus)

    log_dir = f"{os.environ['game24_root']}/{model_pth.split('/')[-1]}/"
    log_dir += f"{start_index}to{end_index}_cot/"
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
        max_length=2048,  ## the prompt is <500 tokens
        max_new_tokens=256,  ## the output is <100 tokens
    )

    batch_cot_game24(
        log_dir=log_dir,
        start_index=start_index,
        end_index=end_index,
        batch_size=batch_size,
        base_model=base_model,
        prompt_name=prompt_name,
        output_name=prompt_name,
        **kwargs,
    )


if __name__ == "__main__":
    fire.Fire(main)
