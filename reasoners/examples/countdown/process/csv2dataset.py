import os
import fire
from pathlib import Path
import pandas as pd
from datasets import load_dataset


def main(
    log_pth: str,
    dataset_prefix: str,
    frac: float,
):

    target_pths = sorted(Path(log_pth).rglob("all_train.csv"))

    for idx, target_pth in enumerate(target_pths):
        target_pth = str(target_pth)
        df = pd.read_csv(target_pth)

        dir_name = os.path.join(os.environ["data_pth"], f"{dataset_prefix}_{idx + 1}")
        os.makedirs(dir_name, exist_ok=True)
        print("\nProcessing", idx + 1, target_pth)
        print("Saving to", dir_name)

        #### now generate correct datasets
        df_correct = df[df["success"] == True]

        train_path = os.path.join(dir_name, "correct_train.csv")
        df_correct.to_csv(train_path, index=False)
        # load_dataset("csv", data_files={"train": train_path})

        #### now generate failed datasets
        df_failed = df[df["success"] == False]
        # downsample to save time for training
        df_failed = df_failed.sample(n=int(frac * 500_000), random_state=42)

        train_path = os.path.join(dir_name, "failed_train.csv")
        df_failed.to_csv(train_path, index=False)
        # load_dataset("csv", data_files={"train": train_path})


if __name__ == "__main__":
    fire.Fire(main)
