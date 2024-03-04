#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
vector service
"""

import os
# import nltk

# work_dir = '/kaggle'
# nltk.data.path.append(os.path.join(work_dir, 'nltk_data'))


from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader, UnstructuredPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from rapidocr_onnxruntime import RapidOCR


class KnowledgeService(object):
    # def __init__(self, config):
    #     self.config = config
    #     self.knowledge_base = None
    #     self.docs_path = self.config.docs_path
    #     self.knowledge_base_path = self.config.knowledge_base_path
    #     self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model_path)

    def __init__(self):
        self.knowledge_base = None
        self.docs_path = '/kaggle/ChatGLM3/docs/'
        self.knowledge_base_path = '/kaggle/ChatGLM3/knowledge_base/'
        # self.embeddings = '/kaggle/text2vec-large-chinese'
        self.embeddings = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese')
    #     看看显存够不够
    #     与这个绝对路径无关


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

    def add_document(self, document_path):
        split_doc = []
        if document_path.endswith('.txt'):
            print(document_path)
            loader = UnstructuredFileLoader(document_path, mode="elements")
            doc = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            split_doc = text_splitter.split_documents(doc)
        elif doc.endswith('.md'):
            print(document_path)
            loader = UnstructuredMarkdownLoader(document_path, mode="elements")
            doc = loader.load()
            markdown_splitter = MarkdownTextSplitter(chunk_size=200, chunk_overlap=20)
            split_doc = markdown_splitter.split_documents(doc)
        elif doc.endswith('.pdf'):
            print(document_path)
            loader = UnstructuredPDFLoader(document_path, mode="elements")
            doc = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            split_doc = text_splitter.split_documents(doc)
        elif doc.endswith('.jpg'):
            print(document_path)
            loader = UnstructuredPDFLoader(document_path, mode="elements")
            docs = self.init_knowledge_base(jpg_file)
            text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            split_doc = text_splitter.create_documents([docs])

        if not self.knowledge_base:
            self.knowledge_base = FAISS.from_documents(split_doc, self.embeddings)
        else:
            self.knowledge_base.add_documents(split_doc)

    #  下面这个函数没有被调用，这个应该是以及转化好的向量知识库保存的位置，可以直接调用，省去转化的步骤
    def load_knowledge_base(self, path):
        if path is None:
            self.knowledge_base = FAISS.load_local(self.knowledge_base_path, self.embeddings)
        else:
            self.knowledge_base = FAISS.load_local(path, self.embeddings)
        return self.knowledge_base


# 这两个函数`init_knowledge_base`和`add_document`都是假定的代码片段，
# 应该是用于处理和存储文档的一部分。下面我将分别解释每个函数的作用以及`docs`和`knowledge_base`的区别。

# 1. `init_knowledge_base`函数的作用:

# 这个函数的目的似乎是初始化一个知识库，它通过加载不同格式的文件（如txt, md, pdf, jpg），
# 并将它们转换成内部的数据结构。这里的步骤通常包括：

# - 遍历一个文档路径(`self.docs_path`)，加载不同格式的文件。
# - 使用不同的加载器(`loader`)根据文件格式（txt, md, pdf, jpg）读取文件内容。
# - 对读取的文件内容进行分割，这里的分割可能是将文档切割成更小的部分以便于之后的处理。
# - 对于图片文件（jpg格式），使用OCR技术提取文本内容。
# - 将所有处理过的文档内容添加到`docs`列表中。

# - 最后将`docs`列表中的内容转换为向量形式，并存储在`self.knowledge_base`对象中，
# 使用的是FAISS库，这是一个高效的相似性搜索和密集向量聚类库。

# 2. `add_document`函数的作用:
# 这个函数的目的是向已经初始化的知识库中添加额外的文档。它处理单个文件（`document_path`参数），
# 并且将该文件分割成小的部分（如果需要），然后添加到知识库中。这个函数的步骤是：

# - 确定传入的`document_path`文件格式。
# - 使用相应的加载器读取和分割文档内容。
# - 如果`self.knowledge_base`对象未初始化，用当前文档初始化它。
# - 如果已初始化，则将新文档的内容添加到现有的知识库中。
#
# `docs`和`knowledge_base`的区别:
# - `docs`是一个临时的列表，用来存储从文件中加载并可能经过分割的原始文档数据。

# - `knowledge_base`是一个对象，它表示构建好的、可以用于检索和分析的知识库。
# 在`init_knowledge_base`函数中，它是通过将`docs`中的文档转换为向量并用FAISS处理来构建的。

# 简而言之，`docs`是原始文档数据的集合，而`knowledge_base`是这些数据经过处理和向量化后，用于快速搜索和检索的结构化形式。`init_knowledge_base`函数用于初始化整个知识库，而`add_document`函数用于向已存在的知识库中添加新的文档。