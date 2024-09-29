import copy

# 对于qwen2和qwen2.5系列模型
IM_START_ID = 151644
IM_END_ID = 151645


def build_prompt(generation_config,
                 tokenizer,
                 query,
                 history=None,
                 system=""):
    if history is None:
        history = []

    # 设置user提问内容的token
    im_start, im_start_tokens = '<|im_start|>', [IM_START_ID]
    im_end, im_end_tokens = '<|im_end|>', [IM_END_ID]

    # 换行符token
    nl_tokens = tokenizer.encode("\n")

    # 用于编码system/user/assistant的一段发言, 格式{role}\n{content}
    def _tokenize_str(role, content):  # 返回元组，下标0是文本，下标1是token ids
        return f"{role}\n{content}", tokenizer.encode(role) + nl_tokens + tokenizer.encode(content)

    # 剩余token数
    if not hasattr(generation_config, 'max_window_size'):
        generation_config.max_window_size = 2048
    left_token_space = generation_config.max_window_size

    # prompt头部: system发言
    system_text_part, system_tokens_part = _tokenize_str("system",
                                                         system)  # system_text_part --> system\nYou are a helpful assistant.
    # 两头加上im_start和im_end
    system_text = f'{im_start}{system_text_part}{im_end}'  # <|im_start|>system\nYou are a helpful assistant.<|im_end|>
    # 对应的token也加上im_start_token和im_end_token
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens
    left_token_space -= len(system_tokens)

    # prompt尾部: user提问内容和assistant引导
    query_text_part, query_tokens_part = _tokenize_str('user', query)
    query_tokens_prefix = nl_tokens + im_start_tokens
    query_tokens_suffix = im_end_tokens + nl_tokens + im_start_tokens + tokenizer.encode('assistant') + nl_tokens

    # query太长截断，根据最长输入token长度限制，保留query_token中的最大部分
    if len(query_tokens_prefix) + len(query_tokens_part) + len(query_tokens_suffix) > left_token_space:
        query_token_len = left_token_space - len(query_tokens_prefix) - len(query_tokens_suffix)
        query_tokens_part = query_tokens_part[:query_token_len]
        query_text_part = tokenizer.decode(query_tokens_part)

    # 拼接得到完整的query_token以及query_text
    query_tokens = query_tokens_prefix + query_tokens_part + query_tokens_suffix
    query_text = f"\n{im_start}{query_text_part}{im_end}\n{im_start}assistant\n"
    left_token_space -= len(query_tokens)

    '''prompt腰部: 历史user+assitant对话
    '''
    history_text, history_tokens = '', []

    # 倒序遍历history信息
    for hist_query, hist_response in reversed(history):
        hist_query_text, hist_query_tokens_part = _tokenize_str("user", hist_query)  # user\n历史提问
        hist_response_text, hist_response_tokens_part = _tokenize_str("assistant", hist_response)  # assistant\n历史回答
        # 生成本轮对话, 两头加上im_start和im_end信息
        cur_history_tokens = nl_tokens + im_start_tokens + hist_query_tokens_part + im_end_tokens + nl_tokens + im_start_tokens + hist_response_tokens_part + im_end_tokens
        cur_history_text = f"\n{im_start}{hist_query_text}{im_end}\n{im_start}{hist_response_text}{im_end}"
        # 储存多轮对话
        if len(cur_history_tokens) <= left_token_space:
            history_text = cur_history_text + history_text
            history_tokens = cur_history_tokens + history_tokens
            left_token_space -= len(cur_history_tokens)
        else:
            break

    # 生成完整Prompt
    # prompt头部：system
    # prompt腰部：history
    # prompt尾部：query
    prompt_str = f'{system_text}{history_text}{query_text}'
    prompt_tokens = system_tokens + history_tokens + query_tokens
    return prompt_str, prompt_tokens


def remove_stop_words(token_ids, stop_words_ids):
    token_ids = copy.deepcopy(token_ids)
    while len(token_ids) > 0:
        # 从最后往前检查，是否有系统停用词
        if token_ids[-1] in stop_words_ids:
            token_ids.pop(-1)
        else:
            break
    return token_ids
