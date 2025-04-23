cd reasoners
export PYTHONPATH=${PWD}:$PYTHONPATH
export data_pth=../data/countdown/corpora

### generating datasets for FT
## the frac controls the negative data size
# log_pth=../data/countdown/inf_logs/Qwen2.5-Math-1.5B; prefix=Qwen2.5-Math-1.5B/cot; frac=0.35
# log_pth=../data/countdown/inf_logs/Qwen2.5-Math-7B; prefix=Qwen2.5-Math-7B/cot; frac=0.5

# python examples/countdown/process/csv2dataset.py --log_pth $log_pth --dataset_prefix $prefix --frac $frac

### data analysis based on the corpora
root_pth=../data/countdown/corpora

# python examples/countdown/process/data_analysis.py --root_pth $root_pth
