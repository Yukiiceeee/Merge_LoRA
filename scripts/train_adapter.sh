export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=1

python ../src/train_adapter.py \
    --use_deepspeed false \
    --model_name_or_path /d2/mxy/Models/Qwen2-7B \
    --domain med \
    --task mc \
    --train_data_path /d2/mxy/TASA/data/data_adapters/med/mc/train.json \
    --eval_data_path /d2/mxy/TASA/data/data_adapters/med/mc/test.json \
    --output_dir /d2/mxy/TASA/models/adapters \
    --peft_type lora \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.1 \
    --model_max_length 512 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --do_train True \
    --do_eval True \
    >> /d2/mxy/TASA/log/train/train_adapter_med_mc.log 2>&1