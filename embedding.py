"""
嵌入模型 - 使用BAAI/bge-small-zh-v1.5进行文本向量化
"""
import os
import sys
import numpy as np
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings

# 配置模型名称和路径
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
VECTOR_DB_PATH = "./dataset_vector_db"
os.environ['HF_HOME'] = './models'
DATA_ROOT_DIR = "/mnt/vepfs/users/data"  # 【请修改】你的大数据库根目录路径

# --- 1. 自动解决网络问题 (关键步骤) ---
# 检测是否在中国环境，如果是，自动设置 HF 镜像，解决模型无法加载问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("🔧 系统配置: 已配置 HF 镜像源，防止模型下载超时。")

class BGEEmbeddings:
    """BGE-Small-ZH 嵌入模型封装"""

    def __init__(self, model_name=MODEL_NAME):
        print(f"⏳ 正在加载 Embedding 模型 ({model_name})...")
        print("   (第一次运行会自动下载模型，约 100MB，请耐心等待)")

        try:
            # 使用 CPU 强制加载以保证稳定性，如果确信有显卡可改为 'cuda'
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'}, 
                encode_kwargs={'normalize_embeddings': True}
            )
            # 添加 embedding_dim 属性，更新为 512
            self.embedding_dim = 512
            print("✅ Embedding 模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载严重失败: {e}")
            print("💡 建议: 请检查网络，或手动下载模型文件夹到本地并修改 MODEL_NAME 为绝对路径。")
            sys.exit(1)

    def embed_query(self, text: str) -> np.ndarray:
        """对单个查询文本进行嵌入"""
        return np.array(self.embeddings.embed_query(text), dtype=np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """对多个文档进行嵌入"""
        return np.array(self.embeddings.embed_documents(texts), dtype=np.float32)
