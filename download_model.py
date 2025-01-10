from modelscope import snapshot_download, AutoTokenizer

model_id = "Qwen/Qwen1.5-0.5B-Chat"

model_dir = snapshot_download(model_id)
