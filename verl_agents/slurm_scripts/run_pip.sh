srun -p Gveval-T --gres=gpu:0 --cpus-per-task=1 -n1 --ntasks-per-node=1 --quotatype=reserved --job-name=pyvision \
pip install flash-attn --no-build-isolation