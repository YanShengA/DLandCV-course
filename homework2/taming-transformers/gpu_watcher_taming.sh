#!/bin/bash

# ================== 配置区 (请根据需要修改) ==================

# 1. 您的最终训练脚本 (run_vqgan.sh)
# 这个监控脚本将会自动调用您已经配置好的 run_vqgan.sh
# 请确保 run_vqgan.sh 脚本与此监控脚本在同一个目录下
TRAINING_SCRIPT_NAME="run_vqgan.sh"

# 2. 闲置阈值 (单位: MiB)
# 当显卡已用显存低于此值时，脚本会认为它是“空闲”的
IDLE_MEM_THRESHOLD=1000

# 3. 轮询间隔 (单位: 秒)
# 每隔多少秒检查一次GPU状态
POLL_INTERVAL=30

# ==========================================================


# 检查 nvidia-smi 命令是否存在
if ! command -v nvidia-smi &> /dev/null
then
    echo "错误: 未找到 'nvidia-smi' 命令。请确保已正确安装 NVIDIA 驱动。"
    exit 1
fi

# 检查您的训练脚本是否存在
if [ ! -f "$TRAINING_SCRIPT_NAME" ]; then
    echo "错误: 未找到训练脚本 '$TRAINING_SCRIPT_NAME'。请确保它与本脚本在同一目录。"
    exit 1
fi


echo "=== Taming-Transformers 全自动 GPU 监控与任务启动脚本 ==="
echo "配置: 显存阈值 < ${IDLE_MEM_THRESHOLD}MiB, 轮询间隔 ${POLL_INTERVAL}s"
echo "监控目标: $TRAINING_SCRIPT_NAME"
echo "------------------------------------------------------------------"

while true; do
    # 初始化找到的空闲GPU ID
    IDLE_GPU_ID=""

    # 使用 nvidia-smi 查询所有GPU的索引和已用显存
    while IFS=',' read -r index mem_used; do
        # 去除 mem_used 可能的前后空格
        mem_used=$(echo "$mem_used" | xargs)
        
        # 判断显存是否低于阈值
        if (( mem_used < IDLE_MEM_THRESHOLD )); then
            # 找到了第一张空闲的物理卡
            IDLE_GPU_ID=$index
            break
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

    # 检查是否找到了空闲GPU
    if [ -n "$IDLE_GPU_ID" ]; then
        echo "" # 换行
        echo "$(date '+%Y-%m-%d %H:%M:%S') - ✅ 成功！发现空闲物理 GPU: $IDLE_GPU_ID"
        
        echo "------------------------------------------------------------------"
        echo "即将使用 GPU $IDLE_GPU_ID 启动训练脚本 '$TRAINING_SCRIPT_NAME'..."
        echo "------------------------------------------------------------------"
        
        # 【核心】执行您的训练脚本，并将找到的 GPU ID 作为第一个参数传进去
        ./${TRAINING_SCRIPT_NAME} ${IDLE_GPU_ID}
        
        # 任务已启动，跳出外部的监控循环
        break
    else
        # 如果没有找到空闲GPU，打印等待信息并休眠
        # 使用 \r 实现单行刷新，避免刷屏
        printf "\r$(date '+%Y-%m-%d %H:%M:%S') - 💤 所有 GPU 正忙... 将在 ${POLL_INTERVAL} 秒后重试。"
        sleep $POLL_INTERVAL
    fi
done

echo "" # 换行
echo "=== 训练任务已启动，监控脚本退出。 ==="