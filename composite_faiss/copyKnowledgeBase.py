import shutil
import os
from config import CONFIG

# 设置目录和embedding基础变量

# 定义源目录和目标目录
# os.makedirs('/kaggle/ChatGLM3/composite_faiss/db')
# source_dir = '/kaggle/ChatGLM3/composite_faiss/db'
source_dir = CONFIG['doc_source']
target_dir = '/kaggle/working'

# 检查目标目录是否存在，如果不存在则创建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 复制整个目录
shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)