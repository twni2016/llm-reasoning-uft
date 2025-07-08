cd reasoners
export PYTHONPATH=${PWD}:$PYTHONPATH

# base_dir=../data/countdown/train_logs/Qwen2.5-Math-1.5B
# base_dir=../data/countdown/train_logs/Qwen2.5-Math-7B
# base_dir=../data/countdown/train_logs_pref/Qwen2.5-Math-1.5B
# base_dir=../data/countdown/train_logs_pref/Qwen2.5-Math-7B

python examples/countdown/process/ft_agg_analysis.py --base_dir $base_dir
