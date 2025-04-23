import os, fire
import pandas as pd
import itertools

with open("examples/game24/metadata/case_ids_train.txt", "r") as f:
    cases_id_train = eval(f.read())
with open("examples/game24/metadata/case_ids_eval.txt", "r") as f:
    cases_id_test = eval(f.read())
with open("examples/game24/metadata/case_ids_ood.txt", "r") as f:
    cases_id_ood = eval(f.read())


def get_results(df, file):
    def succ_rate(df, num_cases, desc):
        any_rate = df.groupby("case_id")["success"].any().sum() / num_cases
        avg_rate = df.groupby("case_id")["success"].mean().sum() / num_cases
        first_rate = df.groupby("case_id")["success"].first().sum() / num_cases
        file.write(
            f"{desc}: {any_rate = :.3f}, {avg_rate = :.3f}, {first_rate = :.3f}\n"
        )
        file.write(
            f"{desc}: {df['success'].sum()} successful paths "
            f"out of all {len(df)} paths\n"
        )

    for cases, desc in zip(
        [cases_id_train, cases_id_test, cases_id_ood],
        ["train_succ", "eval_succ", "ood_succ"],
    ):
        succ_rate(df[df["case_id"].isin(cases)], len(cases), desc)
    file.write("\n")


def train_stats(df, file):
    # report some basic stats of the training dataset
    train_df = df[df["case_id"].isin(cases_id_train)]
    succ_df = train_df[train_df["success"] == True]
    fail_df = train_df[train_df["success"] == False]

    file.write(
        f"succ paths character length:\n {succ_df['text'].apply(len).describe().round(0)}\n"
    )
    file.write(
        f"failed paths character length:\n {fail_df['text'].apply(len).describe().round(0)}\n"
    )
    file.write(
        f"structured paths composes {fail_df['structured'].sum() / len(fail_df):.2f} of failed paths\n"
    )


def main(
    root_pth: str,
):
    dataset_paths = sorted([p for p in os.listdir(root_pth) if "to" in p])
    print(dataset_paths)

    all_dfs = {}

    for dataset_path in dataset_paths:
        file = open(os.path.join(root_pth, dataset_path, "analysis.log"), "w")

        dfs = []
        for trial in sorted(os.listdir(os.path.join(root_pth, dataset_path))):
            if os.path.isdir(os.path.join(root_pth, dataset_path, trial)):
                try:
                    df = pd.read_csv(
                        os.path.join(root_pth, dataset_path, trial, "all.csv")
                    )
                    dfs.append(df)

                    file.write(
                        os.path.join(root_pth, dataset_path, trial, "all.csv") + "\n"
                    )
                    get_results(df, file)
                except:
                    continue
        if len(dfs) == 0:
            file.close()
            continue

        print(os.path.join(root_pth, dataset_path))
        df = pd.concat(dfs)
        df["text"] = df["text"].apply(tuple)
        df = df.drop_duplicates()

        file.write(
            f"\nCombining all datasets in {os.path.join(root_pth, dataset_path)}\n"
        )
        file.write("Only the any() metric and #paths are meaningful\n")
        get_results(df, file)
        train_stats(df, file)

        all_dfs[dataset_path] = df
        file.close()

    file = open(os.path.join(root_pth, "combined.log"), "w")
    keys = list(all_dfs.keys())
    # iterating all combinations (2^N - 1 - N in total)
    for r in range(2, len(keys) + 1):
        for combination in itertools.combinations(keys, r):
            # Combine the selected datasets
            combined_dfs = [all_dfs[key] for key in combination]
            df = pd.concat(combined_dfs)

            # Write the combination details
            file.write(f"Combining the datasets from {combination}:\n")
            file.write("Only the any() metric and #paths are meaningful\n")
            file.write("Results on correct paths before deduplication\n")
            get_results(df, file)

            # Drop duplicates and write results
            df = df.drop_duplicates()
            file.write("Results on correct paths after deduplication\n")
            get_results(df, file)

    file.close()


if __name__ == "__main__":
    fire.Fire(main)
