# 导入需要的库
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from service.chatglm_service import ChatGLMService
from knowledge_service import KnowledgeService


# 定义LangChainApplication类
class LangChainApplication(object):

    def __init__(self):
        self.llm_service = ChatGLMService()

        self.llm_service.load_model(model_name_or_path='/kaggle/chatglm3-6b')

        self.knowledge_service = KnowledgeService()

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


# 创建应用程序实例
application = LangChainApplication()

# 初始化 Streamlit 应用配置
st.set_page_config(
    page_title="ChatGLM3-6B Streamlit Simple Demo",
    page_icon=":robot:",
    layout="wide"
)

# 页面侧边栏控件配置
max_length = st.sidebar.slider("Max Length", 0, 32768, 8192, step=1)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.6, step=0.01)

# 清理会话历史按钮
buttonClean = st.sidebar.button("清理会话历史")
if buttonClean:
    st.session_state.history = []


# 显示历史消息
def show_history():
    if "history" not in st.session_state:
        st.session_state.history = []
    for message in st.session_state.history:
        role = "user" if message["role"] == "user" else "assistant"
        with st.container():
            st.write(f"{role}: {message['content']}")


show_history()

# 用户输入文本
prompt_text = st.text_input("请输入您的问题")
if prompt_text:
    # 添加用户输入到历史记录并显示
    st.session_state.history.append({"role": "user", "content": prompt_text})
    show_history()

    # 调用LangChainApplication的方法获取答案
    response = application.get_knowledeg_based_answer(
        query=prompt_text,
        chat_history=st.session_state.history,
        temperature=temperature,
        top_p=top_p,
        top_k=max_length  # 假设您希望使用 max_length 作为 top_k 参数
    )

    # 显示模型回复
    st.session_state.history.append({"role": "assistant", "content": response})
    show_history()

# 重置输入框
if st.button("重置输入"):
    st.session_state.history = []
    show_history()

# 执行应用程序
st.rerun()
