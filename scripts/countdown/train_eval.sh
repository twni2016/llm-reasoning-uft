# export WANDB_ENTITY=<your account>
export WANDB_DIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
num_train_epochs=2

# model_name=Qwen/Qwen2.5-Math-1.5B; batch_size=16; ga=1
model_name=Qwen/Qwen2.5-Math-7B; batch_size=8; ga=2

model_suffix=${model_name##*/}
log_pth="../data/countdown/train_logs/$model_suffix"
export WANDB_PROJECT="countdown-$model_suffix"

datasets=(
    cot
    # bfs_dfs
    # cot_bfs_dfs
)
lrs=(
    # 0.00001 # 1e-5 only for 1.5B
    # 0.000005 # 5e-6
    # 0.000002 # 2e-6
    0.000001 # 1e-6 only for 7B
)
ul_alphas=(
    # 0.0 # SFT
    # 0.000001 # 1e-6
    # 0.00001 # 1e-5
    0.0001 # 1e-4
    # 0.001 # 1e-3
)
seeds=(
    42 
    # 73 
    # 126
)

eval_dirs=()
cd alignment
for dataset in "${datasets[@]}"; do
    for lr in "${lrs[@]}"; do
        for ul_alpha in "${ul_alphas[@]}"; do
            for seed in "${seeds[@]}"; do
                echo "Running countdown, $model_name, $dataset, lr $lr, ga $ga, alpha $ul_alpha, seed $seed"

                export PYTHONPATH=${PWD}:$PYTHONPATH
                datetime=$(date +'%Y%m%d-%H%M%S')
                output_dir="$log_pth/$dataset/lr-$(printf '%.0e' "$lr")-ul-$(printf '%.0e' "$ul_alpha")/$datetime"
                mkdir -p $output_dir
                eval_dirs+=("$output_dir")
                echo "${eval_dirs[@]}"

                accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml fine_tune/ul.py \
                    "recipes/countdown/$dataset.yaml" --model_name_or_path=$model_name \
                    --dataset_root="../data/countdown/corpora" \
                    --per_device_train_batch_size=$batch_size --num_train_epochs=$num_train_epochs \
                    --learning_rate=$lr --gradient_accumulation_steps=$ga \
                    --ul_loss_type="unlikelihood" --ul_alpha=$ul_alpha --seed=$seed \
                    --output_dir=$output_dir 2>&1 | tee "$output_dir/train.log"
            done
        done
    done
done

cd ../reasoners
export PYTHONPATH=${PWD}:$PYTHONPATH
export game24_root=../data/game24/inf_logs
export cd_root=../data/countdown/inf_logs
MAX_PROCESSES=8

for output_dir in "${eval_dirs[@]}"; do
    while [ $(pgrep -f "python examples/countdown/ft_inference.py" | wc -l) -ge $MAX_PROCESSES ]; do
        sleep 5
    done

    {
        python examples/countdown/ft_inference.py --model_pth $output_dir
        python examples/countdown/process/ft_single_analysis.py --base_dir $output_dir
    } &
    sleep 30
done
cd ..

