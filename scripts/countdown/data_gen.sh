cd reasoners
export PYTHONPATH=${PWD}:$PYTHONPATH
MAX_PROCESSES=8
export cd_root=../data/countdown/inf_logs

# split=val
split=test

# model_pth=Qwen/Qwen2.5-Math-1.5B 
model_pth=Qwen/Qwen2.5-Math-7B

## CoT sampling to generate training data (9 variants)
# top_ps="0.7 0.8 0.9"
# temps="0.5 0.7 1.0"
# for top_p in $top_ps; do
#     for temp in $temps; do
#         while [ $(pgrep -f "base_cot.py" | wc -l) -ge $MAX_PROCESSES ]; do
#             break 3 # just run 8 processes in parallel
#         done

#         # Each run costs ~15-20 hrs
#         echo "Countdown, $model_pth, top_p $top_p, temp $temp"
#         python examples/countdown/base_cot.py --batch_size 10000 --start_index 0 --end_index 500000 \
#             --model_pth $model_pth --n 3 --top_p $top_p --temperature $temp --split train &
#         sleep 40
#     done
# done

### ToT (bfs) (12 variants)
# beam_sizes="5 6 7 8 9 10 11 12 13 14 15 16"
# for beam_size in $beam_sizes; do
#     while [ $(pgrep -f "base_search.py" | wc -l) -ge $MAX_PROCESSES ]; do
#         # break 2 # just run 8 processes in parallel
#         sleep 5 # wait for finish
#     done

#     echo "$model_pth, beam_size $beam_size"
#     python examples/countdown/base_search.py --batch_size -1 --start_index 0 --end_index 1000 \
#         --model_pth $model_pth --split $split --search_algo bfs --n_beam $beam_size &
#     sleep 40
# done

### RAP (mcts) (6 variants)
# w_exps="1.0 2.0 4.0 6.0 8.0 10.0"
# for w_exp in $w_exps; do
#     while [ $(pgrep -f "base_search.py" | wc -l) -ge $MAX_PROCESSES ]; do
#         # break 2 # just run 8 processes in parallel
#         sleep 5 # wait for finish
#     done

#     echo "$model_pth, w_exp $w_exp"
#     python examples/countdown/base_search.py --batch_size -1 --start_index 0 --end_index 1000 \
#         --model_pth $model_pth --split $split --search_algo mcts --w_exp $w_exp &
#     sleep 40
# done


## Getting CoT baseline: manually move the logs to ../init/
## cot n=1 greedy-decoding 
# python examples/countdown/base_cot.py --batch_size -1 --start_index 0 --end_index 1000 \
#     --model_pth $model_pth --top_p 1.0 --temperature 0.0 --split $split --dedup=False
## cot sampling 
# python examples/countdown/base_cot.py --batch_size -1 --start_index 0 --end_index 1000 \
#     --model_pth $model_pth --top_p 0.8 --temperature 0.7 --split $split --dedup=False
