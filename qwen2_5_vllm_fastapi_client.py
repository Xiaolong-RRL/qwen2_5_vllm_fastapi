import requests
import json
import sys


# 清除终端屏幕上的所有内容，并将光标移动到屏幕的左上角，以便显示新的内容
def clear_lines():
    print('\033[2J')


# def clear_lines(num_lines):
#     # 清除前一次的输出
#     for _ in range(num_lines):
#         print('\033[F\033[K', end='')

history = []
line_count = 0  # 用于记录输出的行数
stream = True

while True:
    query = input('问题: ')

    # 调用 api_server
    response = requests.post('http://localhost:8000/', json={
        'query': query,
        'stream': stream,
        'history': history,
    }, stream=True)

    # 流式读取http response body, 按\0分割
    if stream:
        for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode('utf-8'))
                text = data["text"].rstrip('\r\n')  # 确保末尾无换行
                # 清空前一次的内容
                clear_lines()
                # 打印最新内容
                print(text)
    else:  # 非流式输出，整体一次性返回
        data = json.loads(response.text)
        text = data['text'].rstrip('\r\n')  # 确保末尾无换行
        print(text)
        print('\n')

    # 对话历史
    history.append((query, text))
    history = history[-5:]
