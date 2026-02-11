"""
配置文件 - 存储所有系统配置参数
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    model_name: str = "BAAI/bge-small-zh-v1.5"
    device: str = "cpu"  # 或 "cpu"
    max_length: int = 8192
    batch_size: int = 32
    normalize_embeddings: bool = True


@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    index_path: str = "./vector_index"
    chunk_size: int = 1024  # 文本块大小
    chunk_overlap: int = 200  # 文本块重叠
    top_k: int = 10  # 检索返回的文档数量


@dataclass
class LLMConfig:
    """LLM配置"""
    api_base: str = "http://103.242.175.254:20012/v1"  # 自定义API端点
    api_key: str = "sk-audit-RynRlfIcooK6vd8E6dW9iqMJTcFStRmU"  # 自定义API密钥
    model_name: str = "gpt-5-mini"  # 或其他模型名称
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 60


@dataclass
class RAGConfig:
    """RAG系统总配置"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # 数据库路径
    database_root: str = "/mnt/vepfs/users/data"  # 数据库根目录
    readme_pattern: str = "README_*.md"  # README文件匹配模式
    
    # 系统提示词
    system_prompt: str = """你是一个专业的数据库文档问答助手。你的任务是根据提供的文档内容，准确回答用户的问题。

回答原则：
1. 如果问题能在文档中找到明确答案，请严格按照文档内容回答，保持原文的准确性
2. 如果问题涉及多个文档的内容，请综合整理后给出完整答案
3. 如果文档中没有直接答案，但可以基于文档内容推理，请说明这是基于文档的推理
4. 如果问题完全超出文档范围，请明确告知用户该信息不在当前文档中

请始终注明信息来源于哪个数据集的文档。"""


def load_config_from_env() -> RAGConfig:
    """从环境变量加载配置"""
    config = RAGConfig()
    
    # 从环境变量覆盖配置
    if os.getenv("API_BASE"):
        config.llm.api_base = os.getenv("API_BASE")
    if os.getenv("API_KEY"):
        config.llm.api_key = os.getenv("API_KEY")
    if os.getenv("MODEL_NAME"):
        config.llm.model_name = os.getenv("MODEL_NAME")
    if os.getenv("DATABASE_ROOT"):
        config.database_root = os.getenv("DATABASE_ROOT")
    if os.getenv("EMBEDDING_DEVICE"):
        config.embedding.device = os.getenv("EMBEDDING_DEVICE")
        
    return config
