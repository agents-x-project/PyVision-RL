# your wandb access key here...
# wandb login --relogin e43bd719483599eed2767e93e30935b9a286aeba

# the IP and port for your Qwen-2.5-72B-Instruct vllm serving
# export LLM_AS_A_JUDGE_BASE="https://api.claudeshop.top/v1"
# export LLM_AS_A_JUDGE_API_KEY="sk-kBQuM0gvNBhOHmKz43b3iQut01bsOgg8Pv76eMKguu6jvncm"

export LLM_AS_A_JUDGE_BASE="http://10.140.0.231:18901/v1" # 10-140-1-174
# export WANDB_API_KEY: "e43bd719483599eed2767e93e30935b9a286aeba"
# export http_proxy: 'http://zhaoshitian:sk2bGlTOvsIr5zH4omSAe6WUWn4CMaUKSXBmipUqmjLgnjK66yKEHFrTczn4@10.1.20.50:23128/'
# export https_proxy: 'http://zhaoshitian:sk2bGlTOvsIr5zH4omSAe6WUWn4CMaUKSXBmipUqmjLgnjK66yKEHFrTczn4@10.1.20.50:23128/'
export no_proxy='10.140.0.231:18901'

# umber of training nodes
export WORLD_SIZE=1

# config for 7B
bash examples/agent/final_merged_v1v8_thinklite.sh

# config for 32B
# bash examples/agent/final_merged_v1v8_thinklite_32b.sh