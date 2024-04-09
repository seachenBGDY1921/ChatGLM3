import streamlit as st
import os

st.set_page_config(
    page_title="ChatGLM3 Demo",
    page_icon=":robot:",
    layout='centered',
    initial_sidebar_state='expanded',
)


import demo_chat
from enum import Enum

DEFAULT_SYSTEM_PROMPT = '''
请根据知识库中的信息，为用户提供关于苏州旅游的相关知识和建议。可以包括但不限于苏州的著名景点、美食推荐、住宿选择、旅行路线规划以及文化活动等。如果用户询问的问题超出了知识库的范围，或者需要实时数据，请告知用户：抱歉，我无法提供实时旅游信息或超出知识库范围的内容。同时，所有回答都请使用Markdown格式呈现。
'''.strip()

# Set the title of the demo
st.title("ChatGLM3 Demo")

# Add your custom text here, with smaller font size
# st.markdown(
#     "<sub>智谱AI 公开在线技术文档: https://lslfd0slxc.feishu.cn/wiki/WvQbwIJ9tiPAxGk8ywDck6yfnof </sub> \n\n <sub> 更多 ChatGLM3-6B 的使用方法请参考文档。</sub>",
#     unsafe_allow_html=True)

script_dir = os.path.dirname(__file__)

wave_css_path = os.path.join(script_dir, 'wave.css')
with open(wave_css_path) as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


class Mode(str, Enum):
    CHAT = '💬 Chat'


with st.sidebar:
    top_p = st.slider(
        'top_p', 0.0, 1.0, 0.8, step=0.01
    )
    temperature = st.slider(
        'temperature', 0.0, 1.5, 0.95, step=0.01
    )
    repetition_penalty = st.slider(
        'repetition_penalty', 0.0, 2.0, 1.1, step=0.01
    )
    max_new_token = st.slider(
        'Output length', 5, 32000, 256, step=1
    )

    cols = st.columns(2)
    export_btn = cols[0]
    clear_history = cols[1].button("Clear History", use_container_width=True)
    retry = export_btn.button("Retry", use_container_width=True)

    system_prompt = st.text_area(
        label="System Prompt (Only for chat mode)",
        height=300,
        value=DEFAULT_SYSTEM_PROMPT,
    )

prompt_text = st.chat_input(
    'Chat with ChatGLM3!',
    key='chat_input',
)

tab = st.radio(
    'Mode',
    [mode.value for mode in Mode],
    horizontal=True,
    label_visibility='hidden',
)

if clear_history or retry:
    prompt_text = ""

match tab:
    case Mode.CHAT:
        demo_chat.main(
            retry=retry,
            top_p=top_p,
            temperature=temperature,
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_token
        )
    case _:
        st.error(f'Unexpected tab: {tab}')
