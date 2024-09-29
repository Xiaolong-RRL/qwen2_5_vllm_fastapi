# qwen2_5_vllm_fastapi
使用FastAPI+vLLM部署Qwen2.5

## Install
```bash
git clone https://github.com/Xiaolong-RRL/qwen2_5_vllm_fastapi.git
cd qwen2_5_vllm_fastapi
pip install vllm, transformers, fastapi, uvicorn
```

## 启动HTTP服务端
```bash
python qwen2_5_vllm_fastapi_server.py
```

## 启动HTTP客户端
```bash
python qwen2_5_vllm_fastapi_client.py
```

## 启动webui网页聊天
```bash
python qwen2_5_vllm_fastapi_webui_gradio.py
```