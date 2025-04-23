import os
from pathlib import Path
import fire
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid", {"grid.linestyle": "--"})

# To be adjusted
plt.rcParams.update(
    {
        "font.size": 13,  # Default font size for text
        "axes.titlesize": 16,  # Font size for axes titles
        "axes.labelsize": 14,  # Font size for axes labels
        "xtick.labelsize": 13,  # Font size for x-tick labels
        "ytick.labelsize": 13,  # Font size for y-tick labels
        "legend.fontsize": 13,  # Font size for legend
        "figure.titlesize": 16,  # Font size for figure title
        "axes.formatter.useoffset": False,
        "axes.formatter.offset_threshold": 1,
    }
)


def get_cd_results(df, name: str):
    N = 100 if "game24" in name else 1000  # cd4_game24
    if "bfs" or "mcts" in name:
        # search method choose the first path
        return df.groupby("case_id")["success"].first().sum() / N
    else:  # cot
        return df.groupby("case_id")["success"].mean().sum() / N


def get_game24_results(df, name: str):
    with open("examples/game24/metadata/case_ids_ood.txt", "r") as f:
        cases_id_game24 = eval(f.read())
    test_df = df[df["case_id"].isin(cases_id_game24)]
    if "bfs" or "mcts" in name:
        # search method choose the first path
        return test_df.groupby("case_id")["success"].first().sum() / len(
            cases_id_game24
        )
    else:  # cot
        return test_df.groupby("case_id")["success"].mean().sum() / len(cases_id_game24)


def main(
    base_dir: str,
):
    # load the performance from a base LLM using in-context learning (before FT)
    x_col = "Optimization Step"
    model_name = base_dir[base_dir.find("train_logs") :].split("/")[1]
    assert "Qwen" in model_name

    cd_icl_pth = os.path.join(os.environ["cd_root"], "init", model_name)
    game24_icl_pth = os.path.join(os.environ["game24_root"], "init", model_name)

    cd_val_cot_df = pd.read_csv(os.path.join(cd_icl_pth, "all_val_cot.csv"))
    cd_test_cot_df = pd.read_csv(os.path.join(cd_icl_pth, "all_test_cot.csv"))
    cd_test_cot8_df = pd.read_csv(os.path.join(cd_icl_pth, "all_test_cot8.csv"))

    cd_n_beam = 5 if "7B" in model_name else 8
    cd_w_exp = 1 if "7B" in model_name else 2
    cd_bfs_df = pd.read_csv(os.path.join(cd_icl_pth, f"all_test_bfs{cd_n_beam}.csv"))
    cd_mcts_df = pd.read_csv(os.path.join(cd_icl_pth, f"all_test_mcts{cd_w_exp}.csv"))

    game24_cot_df = pd.read_csv(os.path.join(game24_icl_pth, "all_cot.csv"))
    game24_cot20_df = pd.read_csv(os.path.join(game24_icl_pth, "all_cot20.csv"))

    game24_n_beam = 6 if "7B" in model_name else 10
    game24_w_exp = 1 if "7B" in model_name else 2
    game24_bfs_df = pd.read_csv(
        os.path.join(game24_icl_pth, f"all_bfs{game24_n_beam}.csv")
    )
    game24_mcts_df = pd.read_csv(
        os.path.join(game24_icl_pth, f"all_mcts{game24_w_exp}.csv")
    )

    init_results = {
        x_col: 0,
        ## cd
        "val_cot_succ": get_cd_results(cd_val_cot_df, "cot"),
        "test_cot_succ": get_cd_results(cd_test_cot_df, "cot"),
        "test_cot8_succ": get_cd_results(cd_test_cot8_df, "cot8"),
        "test_bfs_succ": get_cd_results(cd_bfs_df, "bfs"),
        "test_mcts_succ": get_cd_results(cd_mcts_df, "mcts"),
        ## game24
        "cd4_game24_cot_succ": get_game24_results(game24_cot_df, "cot"),
        "cd4_game24_cot20_succ": get_game24_results(game24_cot20_df, "cot20"),
        "game24_bfs_succ": get_game24_results(game24_bfs_df, "bfs"),
        "game24_mcts_succ": get_game24_results(game24_mcts_df, "mcts"),
    }
    # print(f"{init_results = }")

    # find all assessment folder recursively
    log_paths = sorted(
        [str(dir) for dir in Path(base_dir).rglob("assessment") if dir.is_dir()]
    )
    print(log_paths)

    for log_pth in log_paths:
        # each log_pth is folder with name of time_string
        metrics = [init_results]

        for ckpt_folder in sorted(Path(log_pth).rglob("checkpoint-*")):

            ckpt_step = int(str(ckpt_folder).split("checkpoint-")[-1])
            entry = {x_col: ckpt_step}

            ## cd results
            for split in ["val", "test"]:
                try:
                    df = pd.read_csv(
                        os.path.join(ckpt_folder, f"all_cd4_{split}_zeroshot.csv")
                    )
                    entry[f"{split}_cot_succ"] = get_cd_results(df, "cot")
                except FileNotFoundError:
                    print(ckpt_folder, split, "cot", "not found")
            try:
                df = pd.read_csv(
                    os.path.join(ckpt_folder, f"all_cd4_test_zeroshot8.csv")
                )
                entry["test_cot8_succ"] = get_cd_results(df, "cot8")
            except FileNotFoundError:
                print(ckpt_folder, "cot8", "not found")

            try:
                df = pd.read_csv(
                    os.path.join(ckpt_folder, f"all_cd4_test_bfs{cd_n_beam}.csv")
                )
                entry["test_bfs_succ"] = get_cd_results(df, "bfs")
            except FileNotFoundError:
                pass
                # print(ckpt_folder, "bfs", "not found")
            except pd.errors.EmptyDataError:  # totally fails
                entry["test_bfs_succ"] = 0.0
            try:
                df = pd.read_csv(
                    os.path.join(ckpt_folder, f"all_cd4_test_mcts{cd_w_exp}.csv")
                )
                entry["test_mcts_succ"] = get_cd_results(df, "mcts")
            except FileNotFoundError:
                pass
                # print(ckpt_folder, "mcts", "not found")
            except pd.errors.EmptyDataError:  # totally fails
                entry["test_mcts_succ"] = 0.0

            ## game-24 results
            for cot_name in ["", "20"]:
                try:
                    df = pd.read_csv(
                        os.path.join(
                            ckpt_folder, f"all_cd4_game24_zeroshot{cot_name}.csv"
                        )
                    )
                    entry[f"cd4_game24_cot{cot_name}_succ"] = get_cd_results(
                        df, f"game24_cot{cot_name}"
                    )
                except FileNotFoundError:
                    print(ckpt_folder, "cd4_game24", f"cot{cot_name}", "not found")

            try:
                df = pd.read_csv(
                    os.path.join(ckpt_folder, f"all_game24_bfs{game24_n_beam}.csv")
                )
                entry["game24_bfs_succ"] = get_game24_results(df, "bfs")
            except FileNotFoundError:
                pass
                # print(ckpt_folder, "game24_bfs", "not found")
            except pd.errors.EmptyDataError:  # totally fails
                entry["game24_bfs_succ"] = 0.0

            try:
                df = pd.read_csv(
                    os.path.join(ckpt_folder, f"all_game24_mcts{game24_w_exp}.csv")
                )
                entry["game24_mcts_succ"] = get_game24_results(df, "mcts")
            except FileNotFoundError:
                pass
                # print(ckpt_folder, "game24_mcts", "not found")
            except pd.errors.EmptyDataError:  # totally fails
                entry["game24_mcts_succ"] = 0.0

            metrics.append(entry)

        metrics = pd.DataFrame(metrics).sort_values(by=x_col, ascending=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

        sns.lineplot(
            data=metrics,
            x=x_col,
            y="val_cot_succ",
            ax=ax1,
            color="red",
            label="val_cot",
        )
        sns.lineplot(
            data=metrics,
            x=x_col,
            y="test_cot_succ",
            ax=ax1,
            color="orange",
            label="test_cot",
        )
        sns.lineplot(
            data=metrics,
            x=x_col,
            y="test_cot8_succ",
            linestyle="dashed",
            ax=ax1,
            color="orange",
            label="test_cot8",
        )
        sns.lineplot(
            data=metrics,
            x=x_col,
            y="test_bfs_succ",
            ax=ax1,
            color="green",
            label="test_bfs",
            alpha=0.7,
        )
        sns.lineplot(
            data=metrics,
            x=x_col,
            y="test_mcts_succ",
            ax=ax1,
            color="magenta",
            label="test_mcts",
            alpha=0.7,
        )

        ax1.set_title("Test on Countdown")
        ax1.set_xlabel(x_col)
        ax1.set_ylabel("Success Rate")
        ax1.legend()

        sns.lineplot(
            data=metrics,
            x=x_col,
            y="cd4_game24_cot_succ",
            ax=ax2,
            color="purple",
            label="cd4_game24_cot",
        )
        sns.lineplot(
            data=metrics,
            x=x_col,
            y="cd4_game24_cot20_succ",
            linestyle="dashed",
            ax=ax2,
            color="purple",
            label="cd4_game24_cot20",
        )

        sns.lineplot(
            data=metrics,
            x=x_col,
            y="game24_bfs_succ",
            ax=ax2,
            color="green",
            label="game24_bfs",
            alpha=0.7,
        )
        sns.lineplot(
            data=metrics,
            x=x_col,
            y="game24_mcts_succ",
            ax=ax2,
            color="magenta",
            label="game24_mcts",
            alpha=0.7,
        )

        ax2.set_title("Test on Game-of-24")
        ax2.set_xlabel(x_col)
        ax2.set_ylabel("Success Rate")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(log_pth, "result.pdf"), dpi=200, bbox_inches="tight")
        plt.close()

        # sort with the highest val_succ
        metrics = metrics.sort_values(by=["val_cot_succ"], ascending=[False])
        metrics = metrics.round(3)
        metrics.to_csv(os.path.join(log_pth, "result.csv"), index=False)


if __name__ == "__main__":
    fire.Fire(main)
