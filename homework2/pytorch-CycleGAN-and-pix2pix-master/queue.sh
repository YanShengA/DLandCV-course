#!/bin/bash

# ================= 配置区 =================

# 1. 目录设置 (建议使用绝对路径，或者确保您在脚本所在目录运行)
TASK_DIR="./task_queue"
RUNNING_DIR="./task_queue/running"
HISTORY_DIR="./task_queue/history"
LOG_FILE="scheduler.log"

# 2. 资源配置
# 每个任务需要占用几张显卡
GPUS_PER_TASK=1

# 【关键】最大允许同时占用的显卡总数 (配额机制)
# 例如：服务器8张卡，限制只用6张。
# 脚本会计算：(正在运行的任务数 * 单任务显卡数) + 新任务显卡数 <= MAX_TOTAL_GPUS
MAX_TOTAL_GPUS=8

# 3. 阈值设置
# 显存低于此值(MiB)视为闲置
IDLE_MEM_THRESHOLD=1000

# 启动任务后的预热等待时间(秒)
# 必须足够长，让 Python 进程加载并占用显存，防止下一轮检测误判
WARMUP_DELAY=30

# 轮询间隔(秒)
POLL_INTERVAL=10

# ===========================================

# 初始化必要的目录
mkdir -p "$TASK_DIR"
mkdir -p "$RUNNING_DIR"
mkdir -p "$HISTORY_DIR"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

log "=== 智能调度器启动 ==="
log "配置: 单任务 $GPUS_PER_TASK 卡, 总配额 $MAX_TOTAL_GPUS 卡, 预热 $WARMUP_DELAY 秒"

while true; do
    # -------------------------------------------------------
    # Step 1: 检查配额 (基于正在运行的任务文件数量)
    # -------------------------------------------------------
    
    # 统计 running 目录下有多少个脚本文件
    NUM_RUNNING=$(ls -1 "$RUNNING_DIR"/*.sh 2>/dev/null | wc -l)
    
    # 计算当前已占用的显卡数
    CURRENT_USAGE=$(( NUM_RUNNING * GPUS_PER_TASK ))
    
    # 检查是否还有余额启动一个新任务
    if (( CURRENT_USAGE + GPUS_PER_TASK > MAX_TOTAL_GPUS )); then
        # 配额已满，不进行后续检查，直接休眠
        # log "配额已满 (已用: $CURRENT_USAGE, 上限: $MAX_TOTAL_GPUS)，等待任务释放..."
        sleep $POLL_INTERVAL
        continue
    fi

    # -------------------------------------------------------
    # Step 2: 检查任务队列
    # -------------------------------------------------------
    
    # 获取最早的一个任务文件
    TASK_FILE=$(ls -1 "$TASK_DIR"/*.sh 2>/dev/null | sort | head -n 1)

    if [ -z "$TASK_FILE" ]; then
        # 队列为空，休眠
        sleep $POLL_INTERVAL
        continue
    fi

    # -------------------------------------------------------
    # Step 3: 检查物理显卡空闲情况 (nvidia-smi)
    # -------------------------------------------------------
    
    IDLE_GPU_INDICES=()
    # 查询显存使用情况
    mapfile -t mem_used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    
    for i in "${!mem_used[@]}"; do
        # 这里不限制 index，只要物理显存空闲，就可以被列入候选
        if (( ${mem_used[$i]} < $IDLE_MEM_THRESHOLD )); then
            IDLE_GPU_INDICES+=($i)
        fi
    done
    
    NUM_IDLE=${#IDLE_GPU_INDICES[@]}

    # -------------------------------------------------------
    # Step 4: 资源匹配与发射任务
    # -------------------------------------------------------

    # 判断物理空闲卡是否足够支持一个任务
    if (( NUM_IDLE >= GPUS_PER_TASK )); then
        TASK_NAME=$(basename "$TASK_FILE")
        log "准备启动任务: $TASK_NAME (当前运行: $NUM_RUNNING, 占用: $CURRENT_USAGE/$MAX_TOTAL_GPUS)"

        # 4.1 选取所需的 GPU ID
        CHOSEN_GPUS=()
        for (( j=0; j<GPUS_PER_TASK; j++ )); do
            CHOSEN_GPUS+=(${IDLE_GPU_INDICES[$j]})
        done
        
        # 拼接 GPU 字符串 (例如 "0,1")
        GPU_STRING=$(IFS=,; echo "${CHOSEN_GPUS[*]}")

        # 4.2 移动任务文件 (状态流转：等待 -> 运行)
        mv "$TASK_FILE" "$RUNNING_DIR/"
        CURRENT_TASK_PATH="$RUNNING_DIR/$TASK_NAME"

        # 4.3 【异步发射】(Fire and Forget)
        (
            # --- 子进程环境 ---
            # 继承当前的虚拟环境和路径
            export CUDA_VISIBLE_DEVICES=$GPU_STRING
            
            log "任务 $TASK_NAME (PID: $$) 已分配 GPU: $GPU_STRING"
            
            # 执行用户的任务脚本
            # 注意：日志重定向应在用户的脚本指令中完成，这里只负责运行
            bash "$CURRENT_TASK_PATH"
            
            EXIT_CODE=$?
            
            # 任务结束后，移动到历史目录
            mv "$CURRENT_TASK_PATH" "$HISTORY_DIR/"
            
            if [ $EXIT_CODE -eq 0 ]; then
                log "任务完成: $TASK_NAME (GPU: $GPU_STRING)"
            else
                log "任务失败: $TASK_NAME (GPU: $GPU_STRING, Exit Code: $EXIT_CODE)"
            fi
            # --- 子进程结束 ---
        ) & 
        
        # 4.4 预热等待
        # 暂停主循环，给刚才启动的 Python 进程一点时间去占用显存
        log "任务发射完毕，等待 $WARMUP_DELAY 秒预热..."
        sleep $WARMUP_DELAY
        
        # 醒来后立即进入下一次循环，利用 continue 再次检查配额和资源
        continue

    else
        # 配额虽然够，但物理显卡不够 (可能被其他非队列程序占用了)
        # log "物理显卡资源不足 (需 $GPUS_PER_TASK, 物理空闲 $NUM_IDLE), 等待..."
        sleep $POLL_INTERVAL
    fi
done