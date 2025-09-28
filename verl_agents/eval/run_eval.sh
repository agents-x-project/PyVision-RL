# srun -p Gveval-T --gres=gpu:0 --cpus-per-task=16 -n1 --ntasks-per-node=1 --quotatype=spot --job-name=pyvision \
python eval_tir_bench.py \
    --model_name deepeyes \
    --api_url http://10.140.60.2:18800/v1 \
    --vstar_bench_path /mnt/petrelfs/zhaoshitian/eaigc3_t_zhaoshitian/data/TIR-Bench/TIR-Bench-V3/TIR_collection_reform_minio3.json \
    --save_path /mnt/petrelfs/zhaoshitian/eaigc3_t_zhaoshitian/data/TIR-Bench \
    --eval_model_name qwen \
    --num_workers 32