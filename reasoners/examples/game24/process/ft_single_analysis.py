import os
from pathlib import Path
import fire
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

with open("examples/game24/metadata/case_ids_train.txt", "r") as f:
    cases_id_train = eval(f.read())
with open("examples/game24/metadata/case_ids_eval.txt", "r") as f:
    cases_id_val = eval(f.read())
with open("examples/game24/metadata/case_ids_ood.txt", "r") as f:
    cases_id_test = eval(f.read())


def get_results(df, name: str):
    def succ_rate(df, num_cases):
        if "bfs" or "mcts" in name:
            # search method choose the first path
            return df.groupby("case_id")["success"].first().sum() / num_cases
        else:  # cot
            return df.groupby("case_id")["success"].mean().sum() / num_cases

    results = {}
    for cases, desc in zip(
        [cases_id_train, cases_id_val, cases_id_test],
        [f"train_{name}_succ", f"val_{name}_succ", f"test_{name}_succ"],
    ):
        results[desc] = succ_rate(df[df["case_id"].isin(cases)], len(cases))
    return results


def main(
    base_dir: str,
):
    # load the performance from a base LLM using in-context learning or search
    model_name = base_dir[base_dir.find("train_logs") :].split("/")[1]
    assert "Qwen" in model_name
    icl_pth = os.path.join(os.environ["game24_root"], "init", model_name)

    # we use few-shot cot results as the baseline for fine-tuned model's zero-shot cot inference
    cot_results = get_results(pd.read_csv(os.path.join(icl_pth, "all_cot.csv")), "cot")
    cot20_results = get_results(
        pd.read_csv(os.path.join(icl_pth, "all_cot20.csv")), "cot20"
    )

    n_beam = 6 if "7B" in model_name else 10
    w_exp = 1 if "7B" in model_name else 2
    bfs_results = get_results(
        pd.read_csv(os.path.join(icl_pth, f"all_bfs{n_beam}.csv")), "bfs"
    )
    mcts_results = get_results(
        pd.read_csv(os.path.join(icl_pth, f"all_mcts{w_exp}.csv")), "mcts"
    )

    x_col = "Optimization Step"
    init_results = {
        x_col: 0,
        **cot_results,
        "test_cot20_succ": cot20_results["test_cot20_succ"],
        "test_bfs_succ": bfs_results["test_bfs_succ"],
        "test_mcts_succ": mcts_results["test_mcts_succ"],
    }

    # find all inference log_folder recursively
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

            try:
                res = pd.read_csv(os.path.join(ckpt_folder, "all_game24_zeroshot.csv"))
                entry.update(get_results(res, "cot"))
            except FileNotFoundError:
                print(ckpt_folder, "all_game24_zeroshot.csv", "not found")
            try:
                res = pd.read_csv(
                    os.path.join(ckpt_folder, "all_game24_zeroshot20.csv")
                )
                entry["test_cot20_succ"] = get_results(res, "cot20")["test_cot20_succ"]
            except FileNotFoundError:
                print(ckpt_folder, "all_game24_zeroshot20.csv", "not found")

            try:
                res = pd.read_csv(
                    os.path.join(ckpt_folder, f"all_game24_bfs{n_beam}.csv")
                )
                entry["test_bfs_succ"] = get_results(res, "bfs")["test_bfs_succ"]
            except FileNotFoundError:
                pass
                # print(ckpt_folder, "bfs not found")
            except pd.errors.EmptyDataError:  # totally fails
                entry["test_bfs_succ"] = 0.0

            try:
                res = pd.read_csv(
                    os.path.join(ckpt_folder, f"all_game24_mcts{w_exp}.csv")
                )
                entry["test_mcts_succ"] = get_results(res, "mcts")["test_mcts_succ"]
            except FileNotFoundError:
                pass
                # print(ckpt_folder, "mcts not found")
            except pd.errors.EmptyDataError:  # totally fails
                entry["test_mcts_succ"] = 0.0

            metrics.append(entry)

        metrics = pd.DataFrame(metrics).sort_values(by=x_col, ascending=True)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        sns.lineplot(
            data=metrics,
            x=x_col,
            y="train_cot_succ",
            ax=ax,
            color="blue",
            label="train_cot",
        )
        sns.lineplot(
            data=metrics,
            x=x_col,
            y="val_cot_succ",
            ax=ax,
            color="red",
            label="val_cot",
        )
        sns.lineplot(
            data=metrics,
            x=x_col,
            y="test_cot_succ",
            ax=ax,
            color="orange",
            label="test_cot",
        )
        sns.lineplot(
            data=metrics,
            x=x_col,
            y="test_cot20_succ",
            linestyle="dashed",
            ax=ax,
            color="orange",
            label="test_cot20",
        )
        sns.lineplot(
            data=metrics,
            x=x_col,
            y="test_bfs_succ",
            ax=ax,
            color="green",
            label="test_bfs",
        )
        sns.lineplot(
            data=metrics,
            x=x_col,
            y="test_mcts_succ",
            ax=ax,
            color="magenta",
            label="test_mcts",
        )

        # ax.set_title(model_name)
        ax.set_xlabel(x_col)
        ax.set_ylabel("Success Rate")

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_pth, "result.pdf"), dpi=200, bbox_inches="tight")
        plt.close()

        metrics = metrics.sort_values(
            by=["val_cot_succ", "train_cot_succ"],
            ascending=[False, True],
        )

        metrics = metrics.round(3)
        metrics.to_csv(os.path.join(log_pth, "result.csv"), index=False)


if __name__ == "__main__":
    fire.Fire(main)
