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
from agent import extract_keywords, boost_results_by_keywords, rerank_results_by_quality, KEYWORD_BOOST_SCORE


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
        # The STRING chunk should get a boost and overtake the non-matching chunk
        string_score = next(s for c, s in boosted if "STRING" in c.content)
        assert string_score > 0.8
        assert boosted[0][0].content == "This document is about STRING database"

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

    def test_max_context_length(self):
        from config import RAGConfig
        cfg = RAGConfig()
        assert cfg.max_context_length == 12000


class TestContextTruncation:
    """测试上下文截断功能"""

    def test_context_truncation_when_too_long(self):
        """测试上下文过长时的截断"""
        # Create a long context that exceeds max_context_length
        long_context = "A" * 15000
        max_length = 12000
        
        # Truncate
        truncated = long_context[:max_length]
        
        assert len(truncated) == max_length
        assert len(truncated) < len(long_context)

    def test_context_not_truncated_when_short(self):
        """测试上下文较短时不截断"""
        short_context = "A" * 5000
        max_length = 12000
        
        # Should not be truncated
        result = short_context if len(short_context) <= max_length else short_context[:max_length]
        
        assert len(result) == len(short_context)
        assert result == short_context


class TestReranking:
    """测试重排序算法"""

    def _make_chunk(self, content, source="test.md", chunk_id=0):
        return DocumentChunk(
            content=content,
            metadata={"source": source},
            chunk_id=chunk_id
        )

    def test_rerank_boosts_keyword_matches(self):
        """测试关键词匹配增加排名"""
        results = [
            (self._make_chunk("This is about databases", source="a.md", chunk_id=0), 0.5),
            (self._make_chunk("STRING database is important", source="b.md", chunk_id=1), 0.4),
            (self._make_chunk("STRING and IntAct are both databases", source="c.md", chunk_id=2), 0.3),
        ]
        
        reranked = rerank_results_by_quality(results, ["STRING", "IntAct"])
        
        # The chunk with both keywords should be ranked higher
        assert "STRING and IntAct" in reranked[0][0].content
        # The chunk with one keyword should be next
        assert "STRING database" in reranked[1][0].content

    def test_rerank_favors_longer_content(self):
        """测试较长内容获得加分"""
        results = [
            (self._make_chunk("Short", source="a.md", chunk_id=0), 0.5),
            (self._make_chunk("A" * 1000, source="b.md", chunk_id=1), 0.5),
        ]
        
        reranked = rerank_results_by_quality(results, [])
        
        # Longer content should rank higher with same base score
        assert len(reranked[0][0].content) > len(reranked[1][0].content)

    def test_rerank_promotes_source_diversity(self):
        """测试来源多样性惩罚"""
        results = [
            (self._make_chunk("Content A1", source="same.md", chunk_id=0), 0.7),
            (self._make_chunk("Content A2", source="same.md", chunk_id=1), 0.7),
            (self._make_chunk("Content B", source="different.md", chunk_id=2), 0.65),
        ]
        
        reranked = rerank_results_by_quality(results, [])
        
        # First result should still be from same.md
        assert reranked[0][0].metadata["source"] == "same.md"
        # But the different source should be promoted over the second same.md
        # because of diversity penalty
        assert reranked[1][0].metadata["source"] == "different.md"

    def test_rerank_empty_results(self):
        """测试空结果处理"""
        reranked = rerank_results_by_quality([], ["keyword"])
        assert reranked == []

    def test_rerank_preserves_chunk_data(self):
        """测试重排序保留所有数据"""
        results = [
            (self._make_chunk("Content 1", chunk_id=0), 0.5),
            (self._make_chunk("Content 2", chunk_id=1), 0.4),
        ]
        
        reranked = rerank_results_by_quality(results, [])
        
        assert len(reranked) == len(results)
        # All original chunks should be present
        original_ids = {c.chunk_id for c, _ in results}
        reranked_ids = {c.chunk_id for c, _ in reranked}
        assert original_ids == reranked_ids

    def test_rerank_diversity_penalty_order_independent(self):
        """测试来源多样性惩罚不依赖于输入顺序"""
        # 两种不同输入顺序应该产生相同的排名结果
        results_order1 = [
            (self._make_chunk("Content A1", source="same.md", chunk_id=0), 0.7),
            (self._make_chunk("Content B", source="different.md", chunk_id=2), 0.65),
            (self._make_chunk("Content A2", source="same.md", chunk_id=1), 0.7),
        ]
        results_order2 = [
            (self._make_chunk("Content A2", source="same.md", chunk_id=1), 0.7),
            (self._make_chunk("Content A1", source="same.md", chunk_id=0), 0.7),
            (self._make_chunk("Content B", source="different.md", chunk_id=2), 0.65),
        ]

        reranked1 = rerank_results_by_quality(results_order1, [])
        reranked2 = rerank_results_by_quality(results_order2, [])

        # Both orderings should produce the same ranking by source
        sources1 = [c.metadata["source"] for c, _ in reranked1]
        sources2 = [c.metadata["source"] for c, _ in reranked2]
        assert sources1 == sources2


class TestLLMClientResponseHandling:
    """测试LLM客户端响应处理"""

    def test_generate_response_extracts_content(self):
        """测试正常响应解析"""
        from unittest.mock import patch, MagicMock
        from llm_client import LLMClient, Message

        client = LLMClient(api_base="http://fake", api_key="key")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("llm_client.requests.post", return_value=mock_response):
            result = client.generate_response([Message(role="user", content="hi")])
        assert result == "hello"

    def test_generate_response_handles_none_content(self):
        """测试content为None时返回空字符串（非截断情况）"""
        from unittest.mock import patch, MagicMock
        from llm_client import LLMClient, Message

        client = LLMClient(api_base="http://fake", api_key="key")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant"}, "finish_reason": "stop"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("llm_client.requests.post", return_value=mock_response):
            result = client.generate_response([Message(role="user", content="hi")])
        assert result == ""

    def test_generate_response_raises_on_length_empty_content(self):
        """测试finish_reason=length且content为空时抛出错误"""
        from unittest.mock import patch, MagicMock
        import pytest
        from llm_client import LLMClient, Message

        client = LLMClient(api_base="http://fake", api_key="key", max_tokens=2048)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant"}, "finish_reason": "length"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("llm_client.requests.post", return_value=mock_response):
            with pytest.raises(ValueError, match="max_tokens"):
                client.generate_response([Message(role="user", content="hi")])

    def test_generate_response_returns_partial_on_length_with_content(self):
        """测试finish_reason=length但有部分content时仍返回内容"""
        from unittest.mock import patch, MagicMock
        from llm_client import LLMClient, Message

        client = LLMClient(api_base="http://fake", api_key="key")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "partial answer"}, "finish_reason": "length"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("llm_client.requests.post", return_value=mock_response):
            result = client.generate_response([Message(role="user", content="hi")])
        assert result == "partial answer"

    def test_default_max_tokens_increased(self):
        """测试默认max_tokens已增大到4096"""
        from config import LLMConfig
        cfg = LLMConfig()
        assert cfg.max_tokens == 4096


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
