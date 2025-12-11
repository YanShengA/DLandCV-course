#!/bin/bash

# ==============================================================================
# --- 用户配置区 ---
# ==============================================================================

# 1. 设置需要使用的GPU数量
NGPUS_REQUIRED=4

# 2. 设置显存阈值 (MiB)
MEMORY_THRESHOLD=1000      

# 3. 检查间隔 (秒)
SLEEP_INTERVAL=60         

# 4. 实验名称与日志
EXP_NAME="Cityscapes_ControlNet_256"
LOG_FILE="./logs/${EXP_NAME}.log"

# 5. 定义端口号
PORT=29515 

# ==============================================================================
# --- 训练命令构建区 ---
# ==============================================================================

construct_command() {
    # 注意：这里直接返回命令字符串，不执行
    echo "export HF_HOME='./hf_cache' && \
    accelerate launch \
    --num_processes ${NGPUS_REQUIRED} \
    --main_process_port ${PORT} \
    --mixed_precision='fp16' \
    controlnet_scripts/train_controlnet.py \
    --pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5' \
    --output_dir='./checkpoints/cityscapes_controlnet_256' \
    --dataset_name='json' \
    --dataset_config_name='default' \
    --train_data_dir='./controlnet_json' \
    --image_column='image' \
    --conditioning_image_column='conditioning_image' \
    --caption_column='text' \
    --resolution=256 \
    --learning_rate=1e-5 \
    --validation_image='/media/HDD0/wzl/mmcls/DLandCV-course/homework2/pytorch-CycleGAN-and-pix2pix-master/datasets/cityscapes_unpaired/valA/1.jpg' \
    --validation_prompt='high quality realistic city street view' \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=10000 \
    --checkpointing_steps=1000 \
    --validation_steps=500 \
    --enable_xformers_memory_efficient_attention \
    --set_grads_to_none"
}

# ==============================================================================
# --- 自动监控逻辑 ---
# ==============================================================================

if ! [[ "$NGPUS_REQUIRED" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误: NGPUS_REQUIRED 必须是一个大于0的整数."
    exit 1
fi

echo "==================================================="
echo "ControlNet 自动监控训练脚本已启动 (纯净版)"
echo "需要GPU数量: ${NGPUS_REQUIRED}"
echo "分辨率: 256x256"
echo "日志文件: ${LOG_FILE}"
echo "==================================================="

while true; do
    free_gpu_ids=()
    # 获取所有GPU ID
    all_gpu_ids=$(nvidia-smi --query-gpu=index --format=csv,noheader)
    current_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$current_time] 正在检查所有GPU状态..."

    for gpu_id in $all_gpu_ids; do
        memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
        # 容错处理
        if ! [[ "$memory_used" =~ ^[0-9]+$ ]]; then
            continue 
        fi

        if [ "$memory_used" -lt "$MEMORY_THRESHOLD" ]; then
            echo "  - GPU $gpu_id: 空闲 (已用 $memory_used MiB)"
            free_gpu_ids+=($gpu_id)
        else
            echo "  - GPU $gpu_id: 占用 (已用 $memory_used MiB)"
        fi
    done

    num_free_gpus=${#free_gpu_ids[@]}
    echo "当前发现 ${num_free_gpus} / ${NGPUS_REQUIRED} 个可用GPU."

    if [ "$num_free_gpus" -ge "$NGPUS_REQUIRED" ]; then
        echo "==================================================="
        echo "资源充足！准备发射任务..."
        
        # 截取前 N 个空闲 ID
        selected_gpus_array=("${free_gpu_ids[@]:0:$NGPUS_REQUIRED}")
        TARGET_GPUS=$(IFS=,; echo "${selected_gpus_array[*]}")
        
        echo "锁定GPU: ${TARGET_GPUS}"
        
        # 构建命令
        TRAIN_CMD=$(construct_command)
        
        # 拼接最终执行串
        FULL_COMMAND="CUDA_VISIBLE_DEVICES=${TARGET_GPUS} nohup bash -c \"${TRAIN_CMD}\" > ${LOG_FILE} 2>&1 &"
        
        echo "将在3秒后执行..."
        sleep 3
        
        # 执行
        eval ${FULL_COMMAND}
        
        if [ $? -eq 0 ]; then
            echo "任务已成功提交！"
            echo "请查看日志: tail -f ${LOG_FILE}"
            echo "脚本退出。"
        else
            echo "错误: 任务提交失败。"
        fi
        break
    else
        echo "资源不足，${SLEEP_INTERVAL} 秒后重试..."
        sleep $SLEEP_INTERVAL
    fi
done