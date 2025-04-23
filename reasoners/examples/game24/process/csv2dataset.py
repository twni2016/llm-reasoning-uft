import os
import fire
from pathlib import Path
import pandas as pd
from datasets import load_dataset

with open("examples/game24/metadata/case_ids_train.txt", "r") as f:
    cases_id_train = eval(f.read())


def main(log_pth: str, dataset_prefix: str):

    target_pths = sorted(Path(log_pth).rglob("all.csv"))
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
        train_df = df_correct[df_correct["case_id"].isin(cases_id_train)]
        train_df.to_csv(train_path, index=False)
        # load_dataset("csv", data_files={"train": train_path})

        #### now generate failed datasets
        df_failed = df[df["success"] == False]

        train_path = os.path.join(dir_name, "failed_train.csv")
        train_df = df_failed[df_failed["case_id"].isin(cases_id_train)]
        train_df.to_csv(train_path, index=False)
        # load_dataset("csv", data_files={"train": train_path})


if __name__ == "__main__":
    fire.Fire(main)
