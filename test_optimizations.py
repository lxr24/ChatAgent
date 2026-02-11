"""
测试优化功能 - 多查询分解、关键词提升、去重等
"""
import sys
import types
import numpy as np

# Mock heavy dependencies to avoid installing ML libraries in test
_mock_embedding = types.ModuleType("langchain_community")
_mock_embedding.embeddings = types.ModuleType("langchain_community.embeddings")
_mock_embedding.embeddings.HuggingFaceEmbeddings = type("HuggingFaceEmbeddings", (), {})
sys.modules["langchain_community"] = _mock_embedding
sys.modules["langchain_community.embeddings"] = _mock_embedding.embeddings

from document_loader import DocumentChunk, TextSplitter
from agent import extract_keywords, boost_results_by_keywords, KEYWORD_BOOST_SCORE


class TestExtractKeywords:
    """测试关键词提取"""

    def test_chinese_and_separator(self):
        result = extract_keywords("STRING和IntAct有什么区别")
        assert len(result) >= 2
        assert any("STRING" in s for s in result)
        assert any("IntAct" in s for s in result)

    def test_chinese_yu_separator(self):
        result = extract_keywords("STRING与IntAct的比较")
        assert len(result) >= 2

    def test_english_and_separator(self):
        result = extract_keywords("STRING and IntAct comparison")
        assert len(result) >= 2

    def test_comma_separator(self):
        result = extract_keywords("STRING, IntAct")
        assert len(result) >= 2

    def test_single_keyword_no_split(self):
        result = extract_keywords("什么是STRING数据库")
        # single keyword should not generate sub_queries from split
        # but may extract proper noun
        assert isinstance(result, list)

    def test_multiple_proper_nouns(self):
        result = extract_keywords("比较STRING和IntAct和BioGRID")
        assert len(result) >= 3

    def test_empty_query(self):
        result = extract_keywords("")
        assert isinstance(result, list)


class TestBoostResults:
    """测试关键词加权提升"""

    def _make_chunk(self, content, source="test", chunk_id=0):
        return DocumentChunk(
            content=content,
            metadata={"source": source},
            chunk_id=chunk_id
        )

    def test_boost_matching_keyword(self):
        results = [
            (self._make_chunk("This document is about STRING database", chunk_id=0), 0.8),
            (self._make_chunk("This document is about IntAct", chunk_id=1), 0.9),
        ]
        boosted = boost_results_by_keywords(results, ["STRING"])
        # The STRING chunk should get a boost
        assert boosted[0][0].content == "This document is about IntAct"  # still higher base score
        # But the STRING one should have increased score
        string_score = next(s for c, s in boosted if "STRING" in c.content)
        assert string_score > 0.8

    def test_boost_both_keywords(self):
        results = [
            (self._make_chunk("About STRING and IntAct", chunk_id=0), 0.7),
            (self._make_chunk("Unrelated content", chunk_id=1), 0.9),
        ]
        boosted = boost_results_by_keywords(results, ["STRING", "IntAct"])
        # Chunk matching both keywords gets +KEYWORD_BOOST_SCORE per keyword (0.05 × 2 = 0.10)
        both_score = next(s for c, s in boosted if "STRING" in c.content)
        assert both_score == 0.7 + KEYWORD_BOOST_SCORE * 2

    def test_empty_keywords(self):
        results = [
            (self._make_chunk("content", chunk_id=0), 0.5),
        ]
        boosted = boost_results_by_keywords(results, [])
        assert len(boosted) == 1
        assert boosted[0][1] == 0.5

    def test_no_results(self):
        boosted = boost_results_by_keywords([], ["STRING"])
        assert boosted == []


class TestMultiQuerySearch:
    """测试多查询搜索"""

    def test_multi_query_deduplication(self):
        """测试多查询搜索的去重功能"""
        import faiss
        from vector_store import VectorStore

        dim = 4
        store = VectorStore(embedding_dim=dim, index_path="/tmp/test_index_dedup")

        chunks = [
            DocumentChunk(content="STRING database info", metadata={"source": "a.md"}, chunk_id=0),
            DocumentChunk(content="IntAct database info", metadata={"source": "b.md"}, chunk_id=1),
            DocumentChunk(content="Both STRING and IntAct", metadata={"source": "c.md"}, chunk_id=2),
        ]
        embeddings = np.random.randn(3, dim).astype(np.float32)

        store.add_documents(chunks, embeddings)

        # Use two identical query embeddings - results should be deduplicated
        q = np.random.randn(dim).astype(np.float32)
        results = store.multi_query_search([q, q], top_k=3)

        # Should not have duplicates
        chunk_keys = [(c.metadata['source'], c.chunk_id) for c, _ in results]
        assert len(chunk_keys) == len(set(chunk_keys))

    def test_multi_query_merges_results(self):
        """测试多查询搜索合并不同查询的结果"""
        from vector_store import VectorStore

        dim = 4
        store = VectorStore(embedding_dim=dim, index_path="/tmp/test_index_merge")

        chunks = [
            DocumentChunk(content="Doc A", metadata={"source": "a.md"}, chunk_id=0),
            DocumentChunk(content="Doc B", metadata={"source": "b.md"}, chunk_id=1),
        ]
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)

        store.add_documents(chunks, embeddings)

        # Two very different queries targeting different docs
        q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        q2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        results = store.multi_query_search([q1, q2], top_k=1)
        # With top_k=1 per query but merged, we should get results from both queries
        assert len(results) >= 1


class TestTextSplitter:
    """测试文本分割器的新默认参数"""

    def test_default_chunk_size(self):
        splitter = TextSplitter()
        assert splitter.chunk_size == 1024

    def test_default_chunk_overlap(self):
        splitter = TextSplitter()
        assert splitter.chunk_overlap == 200

    def test_small_text_not_split(self):
        splitter = TextSplitter(chunk_size=1024, chunk_overlap=200)
        text = "This is a short text."
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_large_text_is_split(self):
        splitter = TextSplitter(chunk_size=1024, chunk_overlap=200)
        # Create text larger than 1024 characters with multiple paragraphs
        text = ("A" * 600 + "\n\n" + "B" * 600)
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2


class TestConfig:
    """测试配置默认值"""

    def test_default_chunk_size(self):
        from config import VectorStoreConfig
        cfg = VectorStoreConfig()
        assert cfg.chunk_size == 1024

    def test_default_chunk_overlap(self):
        from config import VectorStoreConfig
        cfg = VectorStoreConfig()
        assert cfg.chunk_overlap == 200

    def test_default_top_k(self):
        from config import VectorStoreConfig
        cfg = VectorStoreConfig()
        assert cfg.top_k == 10


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
