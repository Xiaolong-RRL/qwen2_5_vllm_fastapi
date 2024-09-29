'''
1. 导入vllm，fastapi，uvicorn等相关python包
'''
import os
import json
import argparse
import uuid  # 生成通用唯一标识符（Universally Unique Identifier，简称 UUID）的功能。UUID 是一种 128 位的数字标识符，通常用于唯一标识信息，例如在分布式系统中标识对象或记录
import uvicorn
from vllm import AsyncEngineArgs, AsyncLLMEngine  # vllm中的异步推理引擎
from vllm.sampling_params import SamplingParams  # vllm中的参数配置
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from qwen2_5_vllm_fastapi_utils import build_prompt, remove_stop_words


# 加载vllm模型以及相关配置参数
def load_vllm(
        model_dir,
        quantization='AWQ',
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype='float16'
):
    global generation_config, tokenizer, stop_words_ids, engine
    # 模型基础配置
    generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.eos_token_id = generation_config.eos_token_id
    # 推理终止词
    stop_words_ids = tokenizer.all_special_ids
    # vLLM基础配置
    args = AsyncEngineArgs(model_dir)
    args.worker_use_ray = False
    args.engine_use_ray = False
    args.tokenizer = model_dir
    args.tensor_parallel_size = tensor_parallel_size
    args.trust_remote_code = True
    args.quantization = quantization
    args.gpu_memory_utilization = gpu_memory_utilization
    args.dtype = dtype
    args.max_num_seqs = 20  # batch最大20条样本
    # 加载模型
    # os.environ['VLLM_USE_MODELSCOPE']='True'
    engine = AsyncLLMEngine.from_engine_args(args)
    return generation_config, tokenizer, stop_words_ids, engine


# 用户停止句匹配
def match_user_stop_words(response_token_ids, user_stop_tokens):
    for stop_tokens in user_stop_tokens:
        if len(response_token_ids) < len(stop_tokens):
            continue
        if response_token_ids[-len(stop_tokens):] == stop_tokens:
            return True  # 命中停止句, 返回True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default="8000")
    parser.add_argument("--quantization", type=str, default='AWQ')
    parser.add_argument("--model-dir", type=str,
                        default="这里要修改为你的模型地址")
    args = parser.parse_args()
    '''
    2. 构建fastapi服务
    '''

    app = FastAPI()

    '''
    3. 使用vllm加载qwen2模型
    '''
    generation_config, tokenizer, stop_words_ids, engine = load_vllm(args.model_dir, args.quantization)


    
    '''
    4. 手动编写接口函数，接收用户输入并且产生输出
    '''
    @app.post("/")
    async def chat(request: Request):
        request = await request.json()  # 读取输入请求中的json段数据

        # 解析request中的参数
        query = request.get('query', None)
        history = request.get('history', [])
        system = request.get('system', 'You are a helpful assistant.')
        stream = request.get("stream", False)
        user_stop_words = request.get("user_stop_words",
                                      [])  # list[str]，用户自定义停止句，例如：['Observation: ', 'Action: ']定义了2个停止句，遇到任何一个都会停止

        if query is None:
            return Response(status_code=502, content='query is empty')

        # 检查是否更新用户停止词
        user_stop_tokens = []
        for words in user_stop_words:
            user_stop_tokens.append(tokenizer.encode(words))

        # 构造prompt
        prompt_text, prompt_tokens = build_prompt(generation_config,
                                                  tokenizer,
                                                  query,
                                                  history=history,
                                                  system=system)

        # 检查构造的prompt
        print(f'prompt_text : {prompt_text}')
        print(f'prompt_tokens : {prompt_tokens}')

        # vLLM请求配置
        sampling_params = SamplingParams(stop_token_ids=stop_words_ids,
                                         early_stopping=False,
                                         top_p=generation_config.top_p,
                                         top_k=-1 if generation_config.top_k == 0 else generation_config.top_k,
                                         temperature=generation_config.temperature,
                                         repetition_penalty=generation_config.repetition_penalty,
                                         max_tokens=generation_config.max_new_tokens)

        # vLLM异步推理（在独立线程中阻塞执行推理，主线程异步等待完成通知）
        request_id = str(uuid.uuid4().hex)  # uuid.uuid4(): 随机生成 UUID，使用伪随机数生成器。
        results_iter = engine.generate(inputs=prompt_text,
                                       sampling_params=sampling_params,
                                       request_id=request_id)

        # 流式返回，即迭代transformer的每一步推理结果并反复返回
        if stream:
            async def streaming_resp():
                async for result in results_iter:
                    # 移除系统提示词
                    token_ids = remove_stop_words(
                        result.outputs[0].token_ids, stop_words_ids
                    )

                    # 返回截止目前的tokens输出
                    text = tokenizer.decode(token_ids)
                    yield (json.dumps({'text': text}) + '\0').encode('utf-8')

                    # 匹配用户停止词,终止推理
                    if match_user_stop_words(token_ids, user_stop_tokens):
                        await engine.abort(request_id)  # 终止vllm后续推理
                        break

            return StreamingResponse(streaming_resp())

        # 不是流式输出，则整体一次性返回
        async for result in results_iter:
            # 移除im_end,eos等系统停止词
            token_ids = remove_stop_words(result.outputs[0].token_ids, stop_words_ids)
            # 返回截止目前的tokens输出
            text = tokenizer.decode(token_ids)
            # 匹配用户停止词,终止推理
            if match_user_stop_words(token_ids, user_stop_tokens):
                await engine.abort(request_id)  # 终止vllm后续推理
                break
        ret = {"text": text}
        return JSONResponse(ret)

    
    '''
    5. 使用uvicorn启动服务
    '''
    # Uvicorn 框架中的一个函数，用于启动一个 ASGI 服务器。
    # Uvicorn 是一个基于 Python 的 ASGI (Asynchronous Server Gateway Interface，异步服务器网关接口) 服务器，
    # 专门用于运行异步框架（如 FastAPI、Starlette 等）
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                workers=1,
                log_level="debug")
