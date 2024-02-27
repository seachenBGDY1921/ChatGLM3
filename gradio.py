

from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate

from service.chatglm_service import ChatGLMService
from ChatGLM3.knowledge_service import KnowledgeService


class LangChainApplication(object):
    # 初始化，加载ChatGLM3模型和本地知识库
    def __init__(self, config):
        self.config = config
        self.llm_service = ChatGLMService()
        self.llm_service.load_model(model_name_or_path=config)
        self.knowledge_service = KnowledgeService(config)

    # 获取大语言模型返回的答案（基于本地知识库查询）
    def get_knowledeg_based_answer(self, query,
                                   history_len=5,
                                   temperature=0.1,
                                   top_p=0.9,
                                   top_k=4,
                                   chat_history=[]):
        # 定义查询的提示模板格式：
        prompt_template = '''
基于以下已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
已知内容:
{context}
问题:
{question}
    '''
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])
        self.llm_service.history = chat_history[-history_len:] if history_len > 0 else []
        self.llm_service.temperature = temperature
        self.llm_service.top_p = top_p

        # 利用预先存在的语言模型、检索器来创建并初始化BaseRetrievalQA类的实例
        knowledge_chain = RetrievalQA.from_llm(
            llm=self.llm_service,
            # 基于本地知识库构建一个检索器，并仅返回top_k的结果
            retriever=self.knowledge_service.knowledge_base.as_retriever(
                search_kwargs={"k": top_k}),
            prompt=prompt)
        # combine_documents_chain的作用是将查询返回的文档内容（page_content）合并到一起作为prompt中context的值
        # 将combine_documents_chain的合并文档内容改为{page_content}

        knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}")

        # 返回结果中是否包含源文档
        knowledge_chain.return_source_documents = True

        # 传入问题内容进行查询
        result = knowledge_chain({"query": query})
        return result

    # 获取大语言模型返回的答案（未基于本地知识库查询）
    def get_llm_answer(self, query):
        result = self.llm_service._call(query)
        return result


# -------------------------------------------
from service.config import LangChainCFG

config = LangChainCFG()
application = LangChainApplication(config)

result = application.get_llm_answer('A25项目要求')
print('\nresult of llm:\n')
print(result)
print('\n#############################################\n')

application.knowledge_service.init_knowledge_base()
result2 = application.get_knowledeg_based_answer('A25项目要求')
print('\n############################################\n')
print('\nresult of knowledge base:\n')
print(result2)

# --------------------------------------------
# 编写Gradio调用函数,开发Gradio界面

from service.config import LangChainCFG
# from service.configuration_chatglm import ChatGLMConfig


# 将文本中的字符转为网页上可以支持的字符，避免被误认为是HTML标签
def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


# 采用流聊天方式（stream_chat）调用ChatGLM模型，使得生成答案有逐字生成的效果
# def predict(input, chatbot, max_length, top_p, temperature, history, past_key_values):
#     chatbot.append((parse_text(input), parse_text(input)))
#     for response, history, past_key_values in model.stream_chat(tokenizer, input, history, past_key_values=past_key_values,
#                                                                 return_past_key_values=True,
#                                                                 max_length=max_length, top_p=top_p,
#                                                                 temperature=temperature):
#         chatbot[-1] = (parse_text(input), parse_text(response))

#         yield chatbot, history, past_key_values

# 在这里定义 application
config = LangChainCFG()
application = LangChainApplication(config)


# 1.#采用流聊天方式（stream_chat）调用本地自定义模型，使得生成答案有逐字生成的效果

def generate_text(input_text, max_length, top_p, temperature, history, past_key_values):
    application.knowledge_service.init_knowledge_base()
    # chatbot.append((parse_text(input_text), parse_text(input_text)))

    response_dict = application.get_knowledeg_based_answer(parse_text(input_text), history_len=5, temperature=0.1,
                                                           top_p=0.9, top_k=4, chat_history=history)

    if 'result' in response_dict and isinstance(response_dict['result'], str):
        result_text = response_dict['result']
        response = parse_text(result_text)

        # 逐字生成
        generated_text = ''
        for char in response:
            generated_text += char
            yield generated_text, history, past_key_values
    else:
        yield None, history, past_key_values


def predict(input_text, chatbot, max_length, top_p, temperature, history, past_key_values):
    # 调用generate_text方法
    chatbot.append((parse_text(input_text), parse_text(input_text)))
    for generated_text, history, past_key_values in generate_text(input_text, max_length, top_p, temperature, history,
                                                                  past_key_values):
        if generated_text:
            result_text = generated_text
            response = parse_text(result_text)
            chatbot[-1] = (parse_text(input_text), response)
            yield chatbot, history, past_key_values
        else:
            yield chatbot, history, past_key_values


# 2.不采用流聊天方式，直接生成答案

# def predict(input_text, chatbot, max_length, top_p, temperature, history, past_key_values):
#     application.knowledge_service.init_knowledge_base()
#     chatbot.append((parse_text(input_text), parse_text(input_text)))


#     response_dict = application.get_knowledeg_based_answer(parse_text(input_text), history_len=5, temperature=0.1, top_p=0.9, top_k=4, chat_history=history)
#     if 'result' in response_dict and isinstance(response_dict['result'], str):
#         result_text = response_dict['result']
#         response = parse_text(result_text)
#         chatbot[-1] = (parse_text(input_text), response)
#         yield chatbot, history, past_key_values
#     else:

#         yield chatbot, history, past_key_values


# 去除输入框的内容
def reset_user_input():
    return gr.update(value='')


# 清除状态
def reset_state():
    return [], [], None


# -----------------------------------------------------
# 运行Gradio界面，运行成功后点击“Running on public URL”后的网页链接即可体验
import gradio as gr

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM3-6B</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10, container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])
    past_key_values = gr.State(None)

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                    [chatbot, history, past_key_values], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)

demo.queue().launch(share=True, inbrowser=True)
