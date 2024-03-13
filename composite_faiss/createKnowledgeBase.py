# coding: utf-8
import os
import glob
from typing import List
# import torch
from multiprocessing import Pool
from tqdm import tqdm
from  langchain_community.document_loaders import (
    CSVLoader, EverNoteLoader,
    PDFMinerLoader, TextLoader,
    UnstructuredEmailLoader, UnstructuredEPubLoader,
    UnstructuredHTMLLoader, UnstructuredMarkdownLoader,
    UnstructuredODTLoader, UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader, UnstructuredExcelLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from config import CONFIG

# 设置目录和embedding基础变量
source_directory = CONFIG['doc_source']
embeddings_model_name = CONFIG['embedding_model']
chunk_size = CONFIG['chunk_size']
chunk_overlap = CONFIG['chunk_overlap']
output_dir = CONFIG['db_source']
k = CONFIG['k']


# Custom document loaders 自定义文档加载
class MyElmLoader(UnstructuredEmailLoader):
    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}), ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}), ".eml": (MyElmLoader, {}), ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}), ".odt": (UnstructuredODTLoader, {}), ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}), ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".xls": (UnstructuredExcelLoader, {}), ".xlsx": (UnstructuredExcelLoader, {}),
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts


def main():
    # Create embeddings
    # print(torch.cuda.is_available())
    # Create and store locally vectorstore
    print("Creating new vectorstore")
    texts = process_documents()
    print(f"Creating embeddings. May take some minutes...")
    embedding_function = SentenceTransformerEmbeddings(model_name=embeddings_model_name)

    # 指定文件夹路径

    folder_save_1 = CONFIG.db_source

    folder_load_2 = CONFIG.db_source

    # 创建或指定数据库文件夹
    os.makedirs(folder_save_1, exist_ok=True)

    # 使用 from_documents 方法创建 VectorStore 实例
    vector_store = FAISS.from_documents(texts, embedding_function)

    # 保存向量数据库到磁盘
    index_name = 'my_index'  # 你可以给索引命名，以便之后加载时使用

    vector_store.save_local(folder_load_2, index_name=index_name)

    # # 从磁盘加载向量数据库，加在client
    # loaded_vector_store = FAISS.load_local(
    #     folder_load_2,
    #     embedding_function,
    #     index_name=index_name,
    #     allow_dangerous_deserialization=True
    # )


if __name__ == "__main__":
    main()