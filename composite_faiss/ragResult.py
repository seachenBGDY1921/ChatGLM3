import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

vector_store_path = '/kaggle/working'
embeddings_model_name = "shibing624/text2vec-base-chinese"
index_name = 'my_index'
embedding_function = SentenceTransformerEmbeddings(model_name=embeddings_model_name)

def load_vector_store(vector_store_path):
    if vector_store_path and os.path.exists(vector_store_path):
        vector_store = FAISS.load_local(
            vector_store_path,
            embedding_function,
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
    return vector_store


def retrieve_documents(query: str):
    # 确保向量存储已加载
    vector_store = load_vector_store(vector_store_path)
    # 使用loaded_vector_store进行文档检索
    results = vector_store.similarity_search_with_score(query)
    return results


query = '苏州市高新区枫桥街道西部白马涧生态园龙池风景区有什么景点'
retrieved_docs = retrieve_documents(query)