import os, re
import pandas as pd
from glob import glob
import fire


def main(base_dir: str):

    # Find all result.csv files recursively
    csv_files = sorted(
        glob(os.path.join(base_dir, "**/assessment/result.csv"), recursive=True)
    )

    aggregated_data = []

    for file in csv_files:
        df = pd.read_csv(file, dtype=str)

        if len(df) < 20:
            print(file, "not enough rows", len(df))
        else:
            print(file, "number of rows", len(df))
            # select the top row, since df is ranked
            top_row = df.iloc[0].copy()
            df["Optimization Step"] = pd.to_numeric(
                df["Optimization Step"], errors="coerce"
            )
            max_step = df["Optimization Step"].max()
            if pd.notnull(max_step) and max_step != 0:
                top_row["Relative Step"] = round(
                    float(top_row["Optimization Step"]) / max_step, 2
                )
            else:
                top_row["Relative Step"] = None

            pattern = (
                rf"{re.escape(base_dir)}/(.+?)/(\d{{8}}-\d{{6}})/assessment/result\.csv"
            )
            match = re.search(pattern, file)
            setting, run_name = match.groups()

            top_row = pd.Series(
                {"setting": setting, "run": run_name, **top_row.to_dict()}
            )  # Add 'run' and 'setting' as first columns

            cols = list(top_row.index)
            cols.insert(2, cols.pop(cols.index("Relative Step")))
            top_row = top_row[cols]

            aggregated_data.append(top_row)

    # Create a DataFrame for the aggregated data
    if aggregated_data:
        aggregated_df = pd.DataFrame(aggregated_data)

        columns_of_interest = [
            "val_cot_succ",
            "test_cot_succ",
            "test_cot8_succ",
            "test_bfs_succ",
            "test_mcts_succ",
            "cd4_game24_cot_succ",
            "cd4_game24_cot20_succ",
            "game24_bfs_succ",
            "game24_mcts_succ",
        ]
        columns_of_interest = [
            col for col in columns_of_interest if col in aggregated_df.columns
        ]

        # Convert these columns to numeric, group by 'setting', and compute mean/std
        stats = (
            aggregated_df.assign(
                **{
                    col: pd.to_numeric(aggregated_df[col], errors="coerce")
                    for col in columns_of_interest
                }
            )
            .groupby("setting")[columns_of_interest]
            .agg(["mean", "std"])  # you can add more aggregations if desired
            .round(3)
            .reset_index()
        )
        stats.columns = ["_".join(col).rstrip("_") for col in stats.columns.values]

        aggregated_df = aggregated_df.merge(stats, on="setting", how="left")

        # Sort the dataframe by `setting` and save to CSV
        aggregated_df.sort_values(by=["setting", "run"], inplace=True)
        output_path = os.path.join(base_dir, "aggregated_result.csv")
        aggregated_df.to_csv(output_path, index=False)
        print(f"Aggregated CSV saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
