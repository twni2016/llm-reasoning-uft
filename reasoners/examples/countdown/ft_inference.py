from lm.vllm_model import VLLM
from lm.utils_gpu import set_cuda_visible_devices
import os, sys
import fire
import pandas as pd
from examples.countdown.base_cot import batch_cot_countdown
from examples.countdown.base_search import batch_search_countdown
from examples.game24.base_search import batch_search_game24


def main(
    model_pth: str,
    batch_size: int = -1,
    num_gpus: int = 1,
    swap_space: int = 32,
    dtype: str = "bfloat16",
):
    log_dir = f"{model_pth}/assessment"  # to appear before checkpoint-*
    os.makedirs(log_dir, exist_ok=True)

    print(sys.argv)
    with open(os.path.join(log_dir, "args.txt"), "w") as f:
        for arg in sys.argv:
            f.write(arg + "\n")

    # use the same gpu, as we will release gpu after each ckpt finishes
    set_cuda_visible_devices(num_gpus)

    try:
        # only evaluate the best ckpt (if exists) to save time
        df = pd.read_csv(os.path.join(log_dir, "result.csv"))
        if len(df) < 18:
            raise ValueError
        ckpt_step = df["Optimization Step"].at[0]
        ckpt_names = [f"checkpoint-{ckpt_step}"]
        print("find the best ckpt", ckpt_names)
        if int(ckpt_step) == 0:
            print("the best ckpt is base model, skip search")
            exit()
        skip_search = False
    except Exception as e:  # does not capture SystemExit
        # evaluate all checkpoints
        ckpt_names = []
        for folder in os.listdir(model_pth):
            if folder.startswith("checkpoint-"):
                ckpt_names.append(folder)
        ckpt_names = sorted(ckpt_names, key=lambda name: int(name.split("-")[1]))
        print(ckpt_names)
        skip_search = True

    for ckpt_name in ckpt_names:

        base_model = VLLM(
            os.path.join(model_pth, ckpt_name),
            num_gpus,
            gpu_mem=0.9,  # 0.95 is too high to cause OOM
            swap_space=swap_space,
            dtype=dtype,
            max_length=2048,
            max_new_tokens=256,
        )

        log_subdir = os.path.join(log_dir, ckpt_name)
        os.makedirs(log_subdir, exist_ok=True)

        print(f"\nLogging to {log_subdir}\n")

        for split in ["val", "test"]:
            if f"all_cd4_{split}_zeroshot.csv" in os.listdir(log_subdir):
                print(f"skip all_cd4_{split}_zeroshot.csv")
            else:
                # CoT w/ greedy decoding on CD
                batch_cot_countdown(
                    log_dir=log_subdir,
                    split=split,
                    prompt_name="zeroshot",
                    output_name="zeroshot",
                    start_index=0,
                    end_index=1000,
                    batch_size=batch_size,
                    base_model=base_model,
                    n=1,
                    temperature=0,
                )
        if f"all_cd4_test_zeroshot8.csv" in os.listdir(log_subdir):
            print(f"skip all_cd4_test_zeroshot8.csv")
        else:
            # CoT sampling on CD
            batch_cot_countdown(
                log_dir=log_subdir,
                split="test",
                prompt_name="zeroshot",
                output_name="zeroshot8",
                start_index=0,
                end_index=1000,
                batch_size=batch_size,
                base_model=base_model,
                n=8,
                temperature=0.7,
                top_p=0.8,
                dedup=False,
            )

        if "all_cd4_game24_zeroshot.csv" in os.listdir(log_subdir):
            print("skip all_cd4_game24_zeroshot.csv")
        else:
            # CoT w/ greedy decoding on game24
            batch_cot_countdown(
                log_dir=log_subdir,
                split="game24",
                prompt_name="zeroshot",
                output_name="zeroshot",
                start_index=900,
                end_index=1000,
                batch_size=batch_size,
                base_model=base_model,
                n=1,
                temperature=0,
            )
        if "all_cd4_game24_zeroshot20.csv" in os.listdir(log_subdir):
            print("skip all_cd4_game24_zeroshot20.csv")
        else:
            # CoT sampling on game24
            batch_cot_countdown(
                log_dir=log_subdir,
                split="game24",
                prompt_name="zeroshot",
                output_name="zeroshot20",
                start_index=900,
                end_index=1000,
                batch_size=batch_size,
                base_model=base_model,
                n=20,
                temperature=0.7,
                top_p=0.8,
                dedup=False,
            )

        n_beam = 6 if "7B" in model_pth else 10
        w_exp = 1 if "7B" in model_pth else 2

        if skip_search or f"all_game24_bfs{n_beam}.csv" in os.listdir(log_subdir):
            print(f"skip all_game24_bfs{n_beam}.csv")
        else:
            # ToT on game24
            print(f"start all_game24_bfs{n_beam}")
            batch_search_game24(
                log_dir=log_subdir,
                start_index=900,  # test set
                end_index=1000,
                batch_size=batch_size,
                base_model=base_model,
                search_algo_name="bfs",
                n_beam=n_beam,
                output_name=f"bfs{n_beam}",
            )

        if skip_search or f"all_game24_mcts{w_exp}.csv" in os.listdir(log_subdir):
            print(f"skip all_game24_mcts{w_exp}.csv")
        else:
            # RAP on game24
            print(f"start all_game24_mcts{w_exp}")
            batch_search_game24(
                log_dir=log_subdir,
                start_index=900,  # test set
                end_index=1000,
                batch_size=batch_size,
                base_model=base_model,
                search_algo_name="mcts",
                w_exp=float(w_exp),
                output_name=f"mcts{w_exp}",
            )

        n_beam = 5 if "7B" in model_pth else 8
        w_exp = 1 if "7B" in model_pth else 2

        if skip_search or f"all_cd4_test_bfs{n_beam}.csv" in os.listdir(log_subdir):
            print(f"skip all_cd4_test_bfs{n_beam}.csv")
        else:
            # ToT on CD
            print(f"start all_cd4_test_bfs{n_beam}")
            batch_search_countdown(
                log_dir=log_subdir,
                split="test",
                start_index=0,  # test set
                end_index=1000,
                batch_size=batch_size,
                base_model=base_model,
                search_algo_name="bfs",
                n_beam=n_beam,
                output_name=f"bfs{n_beam}",
            )

        if skip_search or f"all_cd4_test_mcts{w_exp}.csv" in os.listdir(log_subdir):
            print(f"skip all_cd4_test_mcts{w_exp}.csv")
        else:
            # RAP on CD
            print(f"start all_cd4_test_mcts{w_exp}")
            batch_search_countdown(
                log_dir=log_subdir,
                split="test",
                start_index=0,  # test set
                end_index=1000,
                batch_size=batch_size,
                base_model=base_model,
                search_algo_name="mcts",
                w_exp=float(w_exp),
                output_name=f"mcts{w_exp}",
            )

        base_model.close()


if __name__ == "__main__":
    fire.Fire(main)
