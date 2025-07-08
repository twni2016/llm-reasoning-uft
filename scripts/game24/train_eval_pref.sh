# export WANDB_ENTITY=<your account>
export WANDB_DIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
num_copy_positive=10

# model_name=Qwen/Qwen2.5-Math-1.5B; batch_size=16; ga=1
model_name=Qwen/Qwen2.5-Math-7B; batch_size=8; ga=2

model_suffix=${model_name##*/}
log_pth="../data/game24/train_logs_pref/$model_suffix"
export WANDB_PROJECT="game24-pref-$model_suffix"


datasets=(
    pref_cot
    # pref_tot_rap
    # pref_cot_tot_rap
)
cpo_alphas=(
    # 0.0 # SimPO
    1.0 # CPO-SimPO
)
lrs=(
    # 0.00001 # 1e-5 only for 1.5B
    # 0.000005 # 5e-6
    # 0.000002 # 2e-6
    0.000001 # 1e-6 only for 7B
)
seeds=(
    42 
    # 73 
    # 126 
    # 2024
)

eval_dirs=()
cd alignment
for dataset in "${datasets[@]}"; do
    for cpo_alpha in "${cpo_alphas[@]}"; do
        for lr in "${lrs[@]}"; do
            for seed in "${seeds[@]}"; do
                echo "Running game24, $model_name, $dataset, cpo_alpha $cpo_alpha, lr $lr, ga $ga, seed $seed"

                export PYTHONPATH=${PWD}:$PYTHONPATH
                datetime=$(date +'%Y%m%d-%H%M%S')
                output_dir="$log_pth/$dataset/lr-$(printf '%.0e' "$lr")-cpo-$(printf '%.1f' "$cpo_alpha")/$datetime"
                mkdir -p $output_dir
                eval_dirs+=("$output_dir")
                echo "${eval_dirs[@]}"

                accelerate launch --config_file "recipes/accelerate_configs/deepspeed_zero3.yaml" fine_tune/pref.py \
                    "recipes/game24/$dataset.yaml" --model_name_or_path=$model_name --cpo_alpha=$cpo_alpha \
                    --per_device_train_batch_size=$batch_size --num_copy_positive=$num_copy_positive \
                    --learning_rate=$lr --gradient_accumulation_steps=$ga --seed=$seed \
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
    while [ $(pgrep -f "python examples/game24/ft_inference.py" | wc -l) -ge $MAX_PROCESSES ]; do
        sleep 5
    done

    {
        python examples/game24/ft_inference.py --model_pth $output_dir
        python examples/game24/process/ft_single_analysis.py --base_dir $output_dir
    } &
    sleep 30
done
cd ..
