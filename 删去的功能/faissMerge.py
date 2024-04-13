from config import CONFIG
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os

embeddings_model_name = CONFIG['embedding_model']

embedding_function = SentenceTransformerEmbeddings(model_name=embeddings_model_name)

script_dir = os.path.dirname(os.path.abspath(__file__))
path_to_db = os.path.join(script_dir, 'db')


def load_vector_store(vector_store_path, index_name):
    if vector_store_path and os.path.exists(vector_store_path):
        vector_store = FAISS.load_local(
            vector_store_path,
            embedding_function,
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
    return vector_store


db1 = FAISS.load_local(path_to_db, embedding_function, index_name='my_index', allow_dangerous_deserialization = True)
db2 = FAISS.load_local(path_to_db, embedding_function, index_name='uploadFiles', allow_dangerous_deserialization = True)

db1.merge_from(db2)