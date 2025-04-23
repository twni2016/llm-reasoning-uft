cd reasoners/sos_classic

# Step 1: generating the problem set. 25min
# python generate_problem.py --seed 4 --data_dir ../../data/countdown/inf_logs \
#    --num_samples 500000

# Step 2: dfs costs 6min, bfs costs 8min
# python generate_search.py --seed 4 --data_dir ../../data/countdown/inf_logs \
#     --num_samples 500000 --search dfs
# python generate_search.py --seed 4 --data_dir ../../data/countdown/inf_logs \
#     --num_samples 500000 --search bfs

## Step 3: preprocess to CoT-style paths. 15min
# python process_data.py --search dfs --split train --downsample_size 3
# python process_data.py --search bfs --split train --downsample_size 3
