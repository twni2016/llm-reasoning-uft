cd reasoners
export PYTHONPATH=${PWD}:$PYTHONPATH
export data_pth=../data/game24/corpora

### generating datasets for FT
# log_pth=../data/game24/inf_logs/Qwen2.5-Math-1.5B/0to1362_cot; prefix=Qwen2.5-Math-1.5B/cot
# log_pth=../data/game24/inf_logs/Qwen2.5-Math-1.5B/0to1362_bfs; prefix=Qwen2.5-Math-1.5B/bfs
# log_pth=../data/game24/inf_logs/Qwen2.5-Math-1.5B/0to1362_mcts; prefix=Qwen2.5-Math-1.5B/mcts
# log_pth=../data/game24/inf_logs/Qwen2.5-Math-7B/0to1362_cot; prefix=Qwen2.5-Math-7B/cot
# log_pth=../data/game24/inf_logs/Qwen2.5-Math-7B/0to1362_bfs; prefix=Qwen2.5-Math-7B/bfs
# log_pth=../data/game24/inf_logs/Qwen2.5-Math-7B/0to1362_mcts; prefix=Qwen2.5-Math-7B/mcts

# python examples/game24/process/csv2dataset.py --log_pth $log_pth --dataset_prefix $prefix

### data analysis
# root_pth=../data/game24/inf_logs/Qwen2.5-Math-1.5B
root_pth=../data/game24/inf_logs/Qwen2.5-Math-7B

# python examples/game24/process/data_analysis.py --root_pth $root_pth

