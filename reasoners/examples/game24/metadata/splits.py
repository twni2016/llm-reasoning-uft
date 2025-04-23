import pandas as pd
import random

df = pd.read_csv("examples/game24/metadata/24.csv")

data_size = 900
all_cases = list(range(1, data_size + 1))  # id starts at 1
train_cases = random.sample(all_cases, int(data_size * 0.8))
eval_cases = list(set(all_cases) - set(train_cases))
ood_cases = list(range(data_size + 1, data_size + 101))
assert max(ood_cases) < df["Rank"].max()

with open("examples/game24/metadata/case_ids_train.txt", "w") as file:
    file.write(str(train_cases))
with open("examples/game24/metadata/case_ids_eval.txt", "w") as file:
    file.write(str(eval_cases))
with open("examples/game24/metadata/case_ids_ood.txt", "w") as file:
    file.write(str(ood_cases))
