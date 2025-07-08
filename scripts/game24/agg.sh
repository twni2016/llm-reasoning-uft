cd reasoners
export PYTHONPATH=${PWD}:$PYTHONPATH

# base_dir=../data/game24/train_logs/Qwen2.5-Math-1.5B
# base_dir=../data/game24/train_logs/Qwen2.5-Math-7B
# base_dir=../data/game24/train_logs_pref/Qwen2.5-Math-1.5B
# base_dir=../data/game24/train_logs_pref/Qwen2.5-Math-7B

python examples/game24/process/ft_agg_analysis.py --base_dir $base_dir
