import os, fire
import pandas as pd
import itertools
from pathlib import Path
from tqdm import tqdm

## the logic is a bit different from game 24, since we have data from tabular search
## here we directly read from corpora datasets


def train_stats(df, file):
    if df["success"].all():
        num_succ = len(df.groupby("case_id").head(1))
        file.write(
            f"best-of-n succ rate: {num_succ} / 500000 = {num_succ / 500000}\n\n"
        )

    file.write(f"character length:\n {df['text'].apply(len).describe().round(0)}\n")
    file.write(
        f"structured paths composes {df['structured'].sum() / len(df):.2f} of paths\n"
    )


def main(
    root_pth: str,
):
    dataset_paths = sorted([f for f in Path(root_pth).iterdir() if f.is_dir()])
    file = open(os.path.join(root_pth, "combined.log"), "w")

    ## first analyze successful data
    all_dfs = {}
    for dataset_path in dataset_paths:
        dataset_path = str(dataset_path)
        csv_paths = sorted(Path(dataset_path).rglob("correct_train.csv"))
        print(csv_paths)

        dfs = []
        for csv_path in csv_paths:
            dfs.append(pd.read_csv(csv_path))
        df = pd.concat(dfs, ignore_index=True)
        file.write(f"\n\nSuccessful data: {dataset_path}:\n")
        file.write(f"raw data: {len(df) = }\n")

        df = df.drop_duplicates()
        file.write(f"after dedup: {len(df) = }\n")
        train_stats(df, file)

        all_dfs[dataset_path] = df

    keys = list(all_dfs.keys())
    # iterating all combinations (2^N - 1 - N in total)
    for r in tqdm(range(2, len(keys) + 1)):
        for combination in itertools.combinations(keys, r):
            # Combine the selected datasets
            combined_dfs = [all_dfs[key] for key in combination]
            df = pd.concat(combined_dfs)

            # Write the combination details
            file.write(f"Combining the datasets from {combination}:\n")
            file.write(f"raw data: {len(df) = }\n")

            df = df.drop_duplicates()
            file.write(f"after dedup: {len(df) = }\n")
            train_stats(df, file)

    ## then analyze failed data
    for dataset_path in dataset_paths:
        dataset_path = str(dataset_path)
        csv_paths = sorted(Path(dataset_path).rglob("failed_train.csv"))
        print(csv_paths)

        dfs = []
        for csv_path in csv_paths:
            dfs.append(pd.read_csv(csv_path))
        df = pd.concat(dfs, ignore_index=True)
        file.write(f"\n\nFailed data: {dataset_path}:\n")
        file.write(f"raw data: {len(df) = }\n")

        df = df.drop_duplicates()
        file.write(f"after dedup: {len(df) = }\n")
        train_stats(df, file)

    file.close()


if __name__ == "__main__":
    fire.Fire(main)
