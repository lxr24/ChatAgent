"""
向量存储 - 使用FAISS进行高效向量检索
"""
import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import asdict

from document_loader import DocumentChunk


class VectorStore:
    """FAISS向量存储"""
    
    def __init__(
        self,
        embedding_dim: int,
        index_path: str = "./vector_index",
        use_gpu: bool = False
    ):
        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path)
        self.use_gpu = use_gpu
        
        # 初始化FAISS索引
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.chunk_texts: List[str] = []
        self.metadata_list: List[Dict] = []
        
        # 创建索引目录
        self.index_path.mkdir(parents=True, exist_ok=True)
    
    def _create_index(self, use_ivf: bool = False, nlist: int = 100):
        """创建FAISS索引"""
        if use_ivf and len(self.chunks) > 1000:
            # 使用IVF索引加速大规模检索
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            # 使用简单的内积索引
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            # 转移到GPU
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("FAISS索引已转移到GPU")
    
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray
    ):
        """添加文档到向量存储"""
        if self.index is None:
            self._create_index(use_ivf=len(chunks) > 1000)
        
        # 确保embeddings是正确的格式
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # 归一化向量（用于内积相似度）
        faiss.normalize_L2(embeddings)
        
        # 如果是IVF索引，需要先训练
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            print("训练IVF索引...")
            self.index.train(embeddings)
        
        # 添加向量
        self.index.add(embeddings)
        
        # 存储文档块和元数据
        self.chunks.extend(chunks)
        self.chunk_texts.extend([chunk.content for chunk in chunks])
        self.metadata_list.extend([chunk.metadata for chunk in chunks])
        
        print(f"已添加 {len(chunks)} 个文档块到向量存储")
        print(f"当前索引总文档数: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Tuple[DocumentChunk, float]]:
        """搜索最相似的文档块"""
        if self.index is None or self.index.ntotal == 0:
            print("警告: 向量存储为空")
            return []
        
        # 确保query_embedding是正确的格式
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # 归一化查询向量
        faiss.normalize_L2(query_embedding)
        
        # 搜索
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= score_threshold:
                chunk = self.chunks[idx]
                results.append((chunk, float(score)))
        
        return results
    
    def multi_query_search(
        self,
        query_embeddings: List[np.ndarray],
        top_k: int = 10,
        score_threshold: float = 0.0
    ) -> List[Tuple[DocumentChunk, float]]:
        """使用多个查询向量搜索，合并去重结果"""
        seen_ids = set()
        merged_results = []

        for query_embedding in query_embeddings:
            results = self.search(query_embedding, top_k=top_k, score_threshold=score_threshold)
            for chunk, score in results:
                # 使用 (source, chunk_id) 作为唯一标识进行去重
                chunk_key = (chunk.metadata.get('source', ''), chunk.chunk_id)
                if chunk_key not in seen_ids:
                    seen_ids.add(chunk_key)
                    merged_results.append((chunk, score))

        # 按分数降序排列
        merged_results.sort(key=lambda x: x[1], reverse=True)
        return merged_results

    def search_with_filter(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """带过滤条件的搜索"""
        # 先获取更多结果
        results = self.search(query_embedding, top_k=top_k * 3)
        
        if filter_dict is None:
            return results[:top_k]
        
        # 应用过滤
        filtered_results = []
        for chunk, score in results:
            match = True
            for key, value in filter_dict.items():
                if chunk.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered_results.append((chunk, score))
        
        return filtered_results[:top_k]
    
    def save(self, save_name: str = "index"):
        """保存向量存储到磁盘"""
        # 保存FAISS索引
        index_file = self.index_path / f"{save_name}.faiss"
        
        # 如果是GPU索引，先转回CPU
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_file))
        else:
            faiss.write_index(self.index, str(index_file))
        
        # 保存文档块和元数据
        data = {
            "chunks": [(chunk.content, chunk.metadata, chunk.chunk_id) for chunk in self.chunks],
            "chunk_texts": self.chunk_texts,
            "metadata_list": self.metadata_list,
            "embedding_dim": self.embedding_dim
        }
        
        data_file = self.index_path / f"{save_name}_data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"向量存储已保存到: {self.index_path}")
    
    def load(self, save_name: str = "index") -> bool:
        """从磁盘加载向量存储"""
        index_file = self.index_path / f"{save_name}.faiss"
        data_file = self.index_path / f"{save_name}_data.pkl"
        
        if not index_file.exists() or not data_file.exists():
            print("未找到已保存的索引")
            return False
        
        try:
            # 加载FAISS索引
            self.index = faiss.read_index(str(index_file))
            
            if self.use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
            # 加载文档块和元数据
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            self.chunks = [
                DocumentChunk(content=content, metadata=metadata, chunk_id=chunk_id)
                for content, metadata, chunk_id in data["chunks"]
            ]
            self.chunk_texts = data["chunk_texts"]
            self.metadata_list = data["metadata_list"]
            self.embedding_dim = data["embedding_dim"]
            
            print(f"已加载向量存储，共 {self.index.ntotal} 个文档块")
            return True
            
        except Exception as e:
            print(f"加载向量存储失败: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """获取向量存储统计信息"""
        if self.index is None:
            return {"total_documents": 0}
        
        # 统计每个数据集的文档数
        dataset_counts = {}
        for metadata in self.metadata_list:
            dataset_name = metadata.get("dataset_name", "unknown")
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
        
        return {
            "total_documents": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "dataset_distribution": dataset_counts
        }
    
    def clear(self):
        """清空向量存储"""
        self.index = None
        self.chunks = []
        self.chunk_texts = []
        self.metadata_list = []
        print("向量存储已清空")
