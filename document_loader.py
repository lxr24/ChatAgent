"""
文档加载器 - 负责从数据库文件夹中加载和处理README文档
"""
import os
import re
import glob
from pathlib import Path
from typing import List, Dict, Generator, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Document:
    """文档类"""
    content: str
    metadata: Dict
    
    def __repr__(self):
        return f"Document(source={self.metadata.get('source', 'unknown')}, length={len(self.content)})"


@dataclass
class DocumentChunk:
    """文档块类"""
    content: str
    metadata: Dict
    chunk_id: int
    
    def __repr__(self):
        return f"DocumentChunk(source={self.metadata.get('source', 'unknown')}, chunk_id={self.chunk_id})"


class TextSplitter:
    """文本分割器"""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        将文本分割成块
        优先按段落分割，再按句子分割，最后按字符分割
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        
        # 首先按段落分割
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # 如果当前段落本身超过chunk_size，需要进一步分割
            if len(para) > self.chunk_size:
                # 先保存当前累积的chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # 对长段落按句子分割
                sentences = re.split(r'([。！？.!?])', para)
                temp_chunk = ""
                for i in range(0, len(sentences) - 1, 2):
                    sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
                    if len(temp_chunk) + len(sentence) <= self.chunk_size:
                        temp_chunk += sentence
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence[-self.chunk_overlap:] if len(sentence) > self.chunk_overlap else sentence
                        temp_chunk = sentence
                
                if temp_chunk:
                    current_chunk = temp_chunk[-self.chunk_overlap:] if len(temp_chunk) > self.chunk_overlap else temp_chunk
                    chunks.append(temp_chunk.strip())
                    
            elif len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # 保留重叠部分
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                current_chunk = overlap_text + "\n\n" + para if overlap_text else para
        
        # 添加最后一个chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


class DocumentLoader:
    """文档加载器"""
    
    def __init__(self, database_root: str, readme_pattern: str = "README*.md"):
        self.database_root = Path(database_root)
        self.readme_pattern = readme_pattern
        
    def find_readme_files(self) -> List[Path]:
        """查找所有README文件"""
        readme_files = []
        
        # 遍历所有子文件夹
        for folder in self.database_root.iterdir():
            if folder.is_dir():
                # 查找匹配的README文件
                patterns = [
                    "README_*.md",
                    "README_*.txt", 
                    "README*.md",
                    "README*.txt",
                    "readme_*.md",
                    "readme*.md"
                ]
                
                for pattern in patterns:
                    matches = list(folder.glob(pattern))
                    readme_files.extend(matches)
        
        return list(set(readme_files))  # 去重
    
    def load_document(self, file_path: Path) -> Document:
        """加载单个文档"""
        # 尝试多种编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise ValueError(f"无法读取文件 {file_path}，尝试了所有编码")
        
        # 提取数据集名称
        dataset_name = file_path.parent.name
        
        # 从文件名提取更多信息
        file_stem = file_path.stem
        readme_name = file_stem.replace("README_", "").replace("README", "")
        
        metadata = {
            "source": str(file_path),
            "dataset_name": dataset_name,
            "readme_name": readme_name if readme_name else dataset_name,
            "file_name": file_path.name,
            "folder_path": str(file_path.parent)
        }
        
        return Document(content=content, metadata=metadata)
    
    def load_all_documents(self) -> List[Document]:
        """加载所有文档"""
        readme_files = self.find_readme_files()
        documents = []
        
        print(f"找到 {len(readme_files)} 个README文件")
        
        for file_path in tqdm(readme_files, desc="加载文档"):
            try:
                doc = self.load_document(file_path)
                documents.append(doc)
                print(f"  ✓ 已加载: {file_path.name} (来自 {doc.metadata['dataset_name']})")
            except Exception as e:
                print(f"  ✗ 加载失败 {file_path}: {e}")
        
        return documents


class DocumentProcessor:
    """文档处理器 - 负责文档的分割和预处理"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def process_document(self, document: Document) -> List[DocumentChunk]:
        """处理单个文档，返回文档块列表"""
        chunks = self.splitter.split_text(document.content)
        
        doc_chunks = []
        for i, chunk_content in enumerate(chunks):
            chunk = DocumentChunk(
                content=chunk_content,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                },
                chunk_id=i
            )
            doc_chunks.append(chunk)
        
        return doc_chunks
    
    def process_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """处理多个文档"""
        all_chunks = []
        
        for doc in tqdm(documents, desc="分割文档"):
            chunks = self.process_document(doc)
            all_chunks.extend(chunks)
            
        print(f"共生成 {len(all_chunks)} 个文档块")
        return all_chunks


def create_document_summary(documents: List[Document]) -> str:
    """创建文档摘要"""
    summary = "已加载的数据集文档：\n"
    for doc in documents:
        summary += f"- {doc.metadata['dataset_name']}: {doc.metadata['file_name']}\n"
    return summary
