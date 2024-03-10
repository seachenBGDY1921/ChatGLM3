#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
vector service
"""

import os
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader, UnstructuredPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from rapidocr_onnxruntime import RapidOCR


class KnowledgeService(object):

    def __init__(self):
        self.knowledge_base = None
        self.docs_path = '/kaggle/ChatGLM3/docs/'
        self.knowledge_base_path = '/kaggle/ChatGLM3/knowledge_base/'
        self.embeddings = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese')

    def init_knowledge_base(self):
        """
        初始化本地知识库向量
        """
        print('\n#####init_knowledge_base#####\n')
        docs = []
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        markdown_splitter = MarkdownTextSplitter(chunk_size=200, chunk_overlap=20)
        for doc in os.listdir(self.docs_path):
            if doc.endswith('.txt'):
                print(doc)
                loader = UnstructuredFileLoader(f'{self.docs_path}/{doc}', mode="elements")
                doc = loader.load()
                split_doc = text_splitter.split_documents(doc)
                docs.extend(split_doc)
            elif doc.endswith('.md'):
                print(doc)
                loader = UnstructuredMarkdownLoader(f'{self.docs_path}/{doc}', mode="elements")
                doc = loader.load()
                split_doc = markdown_splitter.split_documents(doc)
                docs.extend(split_doc)
            elif doc.endswith('.pdf'):
                print(doc)
                loader = UnstructuredPDFLoader(f'{self.docs_path}/{doc}', mode="elements")
                doc = loader.load()
                split_doc = markdown_splitter.split_documents(doc)
                docs.extend(split_doc)
            elif doc.endswith('.jpg'):
                print(doc)
                ocr = RapidOCR()
                result, _ = ocr(f'{self.docs_path}/{doc}')
                img_docs = ""
                if result:
                    ocr_result = [line[1] for line in result]
                    img_docs += "\n".join(ocr_result)
                split_docs = text_splitter.create_documents([img_docs])
                docs.extend(split_docs)

        # 这里调用出问题，
        self.knowledge_base = FAISS.from_documents(docs, self.embeddings)

    def search_knowledge_base(self, query, top_k=5):


        # 计算查询文本的向量表示
        query_vector = self.embeddings.embed_query(text=query)

        # 将查询向量转换为适合FAISS搜索的格式
        query_vector = np.array(query_vector).astype("float32").reshape(1, -1)

        # 使用FAISS进行搜索
        D, I = self.knowledge_base.search(query_vector, top_k)

        # 解析搜索结果
        results = []
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            result = {
                'rank': i + 1,
                'distance': distance,
                'document_id': idx,
            }
            results.append(result)

        return results

