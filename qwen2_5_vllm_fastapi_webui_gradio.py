import gradio as gr
import requests
import json

MAX_HISTORY_LEN = 50


def chat_streaming(query, history):
    # 调用api_server
    response = requests.post('http://localhost:8000', json={
        'query': query,
        'stream': True,
        'history': history
    }, stream=True)

    # 流式读取http response body, 按\0分割
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode('utf-8'))
            text = data["text"].rstrip('\r\n')  # 确保末尾无换行
            yield text


# with gr.Blocks(css='.qwen2.5-logo img {height:100px; width:300px; margin:0 auto;}') as app:
with gr.Blocks() as app:
    gr.Markdown("""\
<p align="center"><img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/assets/logo/qwen2.5_logo.png" style="height: 120px"/><p>""")
    gr.Markdown(
        """\
<center><font size=3>This WebUI is based on Qwen2.5-Instruct, developed by Alibaba Cloud. \
(本WebUI基于Qwen2.5-Instruct打造，实现聊天机器人功能。)</center>"""
    )
    #     with gr.Row():
    #         logo_img = gr.Image('qwen2_5.png', elem_classes='qwen2.5-logo')
    with gr.Row():
        chatbot = gr.Chatbot(label='通义千问Qwen2.5-7B-Instruct-AWQ')
    with gr.Row():
        query_box = gr.Textbox(label='提问', autofocus=True, lines=2)
    with gr.Row():
        empty_btn = gr.ClearButton([query_box, chatbot], value='🧹 Clear History (清除历史)')
        submit_btn = gr.Button(value='🚀 Submit (发送)', variant="primary")


    def chat(query, history):
        for response in chat_streaming(query, history):
            yield '', history + [(query, response)]
        history.append((query, response))
        while len(history) > MAX_HISTORY_LEN:
            history.pop(0)


    # 提交query
    submit_btn.click(chat, [query_box, chatbot], [query_box, chatbot], show_progress=True)

if __name__ == "__main__":
    app.queue(200)  # 请求队列
    app.launch(server_name='0.0.0.0', max_threads=500)  # 线程池
