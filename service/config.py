#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
ChatGLM service
"""

import os


class LangChainCFG:
    work_dir = '/kaggle'

    llm_model_name = 'model'
    llm_model_path = os.path.join(work_dir, llm_model_name)

    embedding_model_name = 'text2vec-large-chinese'
    embedding_model_path = os.path.join(work_dir, embedding_model_name)

    docs_path = os.path.join('..', 'docs')
    knowledge_base_path = os.path.join('..', 'docs')