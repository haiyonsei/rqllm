#!/bin/bash

# 실행할 커맨드 목록
#20250510/231729

#./brlv3.py --enc-width [2048, 1024, 512] --dec-width [2048, 1024, 512] --trials 3 # 20250511/105736
# (μ=2.667e-04, σ=2.813e-05)

# binomial std
# manual learning rate change 
# no ternary option
# fixed connections bug


#./autoencoder.py --enc-width [1024, 1024, 1024] --dec-width [1024, 1024, 1024] --trials 5 --epochs 400 --noise-start 0.1 --noise-end 0.01 --noise-sched linear # 20250512/235337
#float16→float8_e3m4→float32 MSE : 1.905e-03
#float16→float8_e4m3→float32 MSE : 7.590e-04
#float16→float8_e5m2→float32 MSE : 3.007e-03
#LOGIC 16→8→16 final MSEs : 2.925e-04, 2.701e-04, 1.724e-04, 2.461e-04, 3.071e-04  (μ=2.577e-04, σ=4.735e-05)

#./autoencoder.py --enc-width [1024, 1024, 1024] --dec-width [1024, 1024, 1024] --trials 5 --epochs 400 --noise-start 0.1 --noise-end 0.01 --noise-sched linear --no-ternary # 20250513/214408
#float16→float8_e3m4→float32 MSE : 1.905e-03
#float16→float8_e4m3→float32 MSE : 7.590e-04
#float16→float8_e5m2→float32 MSE : 3.007e-03
#LOGIC 16→8→16 final MSEs : 2.206e-04, 1.920e-04, 7.921e-04, 1.492e-04, 1.969e-04  (μ=3.102e-04, σ=2.421e-04)
#Results saved to: 20250513/214408/logic_enc_width[1024, 1024, 1024]_dec_width[1024, 1024, 1024]_lr0p01.json

#commands=(
#    "python3 ./autoencoder.py --enc-width '[1024, 1024, 1024]' --dec-width '[1024, 1024, 1024]' --trials 5 --encoder-noise-prob 0.03 --decoder-noise-prob 0.03  --epochs 400"
#    "python3 ./autoencoder.py --enc-width '[2048, 2048, 2048]' --dec-width '[2048, 2048, 2048]' --trials 5 --encoder-noise-prob 0.05 --decoder-noise-prob 0.05  --epochs 400"
#    "python3 ./autoencoder.py --enc-width '[1024, 1024, 1024]' --dec-width '[1024, 1024, 1024]' --trials 5 --encoder-noise-prob 0.01 --decoder-noise-prob 0.01  --epochs 400"
#    "python3 ./autoencoder.py --enc-width '[2048, 2048, 2048]' --dec-width '[2048, 2048, 2048]' --trials 5 --encoder-noise-prob 0.03 --decoder-noise-prob 0.03  --epochs 400"
#    "python3 ./autoencoder.py --enc-width '[1024, 1024, 1024]' --dec-width '[1024, 1024, 1024]' --trials 5 --encoder-noise-prob 0.03 --decoder-noise-prob 0.03  --epochs 400 --lr 0.1"
#    "python3 ./autoencoder.py --enc-width '[1024, 1024, 1024]' --dec-width '[1024, 1024, 1024]' --trials 5 --encoder-noise-prob 0.03 --decoder-noise-prob 0.03  --epochs 400 --lr 0.05"
#    "python3 ./autoencoder.py --enc-width '[1024, 1024, 1024]' --dec-width '[1024, 1024, 1024]' --trials 5 --encoder-noise-prob 0.03 --decoder-noise-prob 0.03  --epochs 400 --k-history 1"
#    "python3 ./autoencoder.py --enc-width '[1024, 1024, 1024]' --dec-width '[1024, 1024, 1024]' --trials 5 --encoder-noise-prob 0.03 --decoder-noise-prob 0.03  --epochs 400 --no-ternary"
#    "python3 ./autoencoder.py --enc-width '[1024, 1024, 1024]' --dec-width '[1024, 1024, 1024]' --trials 5 --encoder-noise-prob 0.03 --decoder-noise-prob 0.03  --epochs 400 --connections 'random'"
#)


commands=(
"python3 ./train_codebooks.py --mode qwk"
"python3 ./train_codebooks.py --mode rq --L 1"
"python3 ./train_codebooks.py --mode rq --L 2"
"python3 ./train_codebooks.py --mode rq --L 4"
"python3 ./train_codebooks.py --mode rq --L 8"
"python3 ./train_codebooks.py --mode pq --n_subvec 1"
"python3 ./train_codebooks.py --mode pq --n_subvec 2"
"python3 ./train_codebooks.py --mode pq --n_subvec 4"
"python3 ./train_codebooks.py --mode pq --n_subvec 8"

)

# 사용 가능한 GPU ID들
gpus=(0 1)
num_gpus=${#gpus[@]}

# 현재 실행 중인 PID 및 로그 파일 경로 배열
pids=()
log_files=()

# 로그 디렉토리 준비
log_dir="logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"

# main loop
for cmd in "${commands[@]}"; do
    while true; do
        for i in "${!gpus[@]}"; do
            gpu_id="${gpus[$i]}"
            
            # 해당 GPU가 비어있는 경우
            if [ -z "${pids[$i]}" ] || ! kill -0 "${pids[$i]}" 2>/dev/null; then
                log_file="$log_dir/gpu${gpu_id}_$(date +%s).log"
                echo "[GPU $gpu_id] Launching: $cmd"
                echo "Logging to $log_file"
                
                CUDA_VISIBLE_DEVICES="$gpu_id" bash -c "$cmd" > "$log_file" 2>&1 &
                pids[$i]=$!
                log_files+=("$log_file")
                sleep 1
                break 2
            fi
        done
        sleep 1
    done
done

# 모든 프로세스가 끝날 때까지 대기
for pid in "${pids[@]}"; do
    if [ -n "$pid" ]; then
        wait "$pid"
    fi
done

# 모든 로그 합치기
final_log="${log_dir}/combined_output.log"
echo "🔗 Merging logs into $final_log"
cat "${log_files[@]}" > "$final_log"

echo "✅ All jobs completed. Logs saved in $log_dir"

