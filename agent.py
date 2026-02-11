"""
主入口文件 - 启动RAG Agent
"""
import re
import logging
from config import load_config_from_env
from document_loader import DocumentLoader, DocumentProcessor, create_document_summary
from embedding import BGEEmbeddings
from vector_store import VectorStore
from llm_client import LLMClient, Message

def setup_logging():
    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建一个文件处理器，将日志写入文件
    file_handler = logging.FileHandler('agent.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # 将处理器添加到logger
    logger.addHandler(file_handler)


KEYWORD_BOOST_SCORE = 0.05
DIVERSITY_PENALTY = 0.02  # 来源重复惩罚
CONTENT_LENGTH_BONUS = 0.01  # 内容长度奖励（较长内容通常更完整）


def extract_keywords(query):
    """从查询中提取关键词，用于生成多个子查询"""
    sub_queries = []

    # 支持的分隔符：英文逗号、中文逗号、英文"and"、中文"和"/"与"/"以及"
    parts = re.split(r'[,，]|\s+and\s+|\s*和\s*|\s*与\s*|\s*以及\s*', query)

    # 匹配大写开头的英文专有名词（如 "STRING"、"IntAct"、"BioGRID"）
    proper_nouns = re.findall(r'\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*\b', query)

    if len(parts) > 1:
        for part in parts:
            part = part.strip()
            if part:
                sub_queries.append(part)

    if proper_nouns and len(proper_nouns) > 1:
        for noun in proper_nouns:
            noun = noun.strip()
            if noun and noun not in sub_queries:
                sub_queries.append(noun)

    return sub_queries


def boost_results_by_keywords(results, keywords):
    """根据关键词对检索结果进行加权提升"""
    if not keywords:
        return results

    boosted = []
    for chunk, score in results:
        content_lower = chunk.content.lower()
        boost = 0.0
        for kw in keywords:
            if kw.lower() in content_lower:
                boost += KEYWORD_BOOST_SCORE
        boosted.append((chunk, score + boost))

    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted


def rerank_results_by_quality(results, keywords):
    """
    重排序算法：根据多个质量因素对检索结果进行重排序
    
    质量因素包括：
    1. 相似度得分（已有）
    2. 关键词匹配度
    3. 内容完整性（长度）
    4. 来源多样性（避免同一来源过度集中）
    
    Args:
        results: List[(DocumentChunk, score)] 检索结果列表
        keywords: List[str] 关键词列表
        
    Returns:
        重排序后的结果列表
    """
    if not results:
        return results
    
    reranked = []
    source_counts = {}  # 跟踪每个来源已经使用的次数
    
    for chunk, score in results:
        quality_score = score
        
        # 1. 关键词匹配加分
        content_lower = chunk.content.lower()
        keyword_matches = 0
        for kw in keywords:
            if kw.lower() in content_lower:
                keyword_matches += 1
        quality_score += keyword_matches * KEYWORD_BOOST_SCORE
        
        # 2. 内容长度加分（较长的内容通常更完整，但有上限）
        content_length_factor = min(len(chunk.content) / 1000, 1.0)  # 归一化到0-1
        quality_score += content_length_factor * CONTENT_LENGTH_BONUS
        
        # 3. 来源多样性（同一来源出现次数越多，惩罚越大）
        source = chunk.metadata.get('source', 'unknown')
        source_count = source_counts.get(source, 0)
        quality_score -= source_count * DIVERSITY_PENALTY
        source_counts[source] = source_count + 1
        
        reranked.append((chunk, quality_score))
    
    # 按质量分数降序排序
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    logging.info(f"重排序完成，共{len(reranked)}个结果")
    
    return reranked


def main():
    setup_logging()

    try:
        # 加载配置
        config = load_config_from_env()

        # 初始化组件
        document_loader = DocumentLoader(config.database_root, config.readme_pattern)
        document_processor = DocumentProcessor(
            chunk_size=config.vector_store.chunk_size,
            chunk_overlap=config.vector_store.chunk_overlap
        )
        embedding_model = BGEEmbeddings(
            model_name=config.embedding.model_name
        )
        vector_store = VectorStore(
            embedding_dim=embedding_model.embedding_dim,
            index_path=config.vector_store.index_path
        )
        llm_client = LLMClient(
            api_base=config.llm.api_base,
            api_key=config.llm.api_key,
            model_name=config.llm.model_name,
            max_tokens=config.llm.max_tokens,
            temperature=config.llm.temperature,
            timeout=config.llm.timeout
        )

        # 加载文档
        documents = document_loader.load_all_documents()
        document_chunks = document_processor.process_documents(documents)
        embeddings = embedding_model.embed_documents([chunk.content for chunk in document_chunks])

        # 构建向量存储
        vector_store.add_documents(document_chunks, embeddings)
        vector_store.save()

        # 启动交互
        while True:
            query = input("用户问题: ")
            if query.lower() in ["exit", "quit"]:
                break

            # 查询向量存储 - 多查询检索
            sub_queries = extract_keywords(query)

            if len(sub_queries) > 1:
                # 对每个子查询分别生成向量
                query_embeddings = [embedding_model.embed_query(sq) for sq in sub_queries]
                # 加入原始完整查询的向量
                query_embeddings.insert(0, embedding_model.embed_query(query))
                results = vector_store.multi_query_search(query_embeddings, top_k=config.vector_store.top_k)
            else:
                query_embedding = embedding_model.embed_query(query)
                results = vector_store.search(query_embedding, top_k=config.vector_store.top_k)

            # 使用重排序算法提升结果质量
            keywords = sub_queries if sub_queries else [query]
            results = rerank_results_by_quality(results, keywords)

            # 构建上下文 - 去重
            seen_contents = set()
            context_parts = []
            for chunk, score in results:
                if chunk.content not in seen_contents:
                    seen_contents.add(chunk.content)
                    context_parts.append(f"来源: {chunk.metadata['source']}\n内容: {chunk.content}")
            context = "\n\n".join(context_parts)

            # 记录原始上下文长度
            original_context_length = len(context)
            logging.info(f"上下文字符数: {original_context_length}")

            # 如果上下文过长，进行截断
            if original_context_length > config.max_context_length:
                # 截断上下文，保留前面的内容（通常相关性更高）
                context = context[:config.max_context_length]
                logging.warning(f"上下文过长（{original_context_length}字符），已截断至{config.max_context_length}字符")
                print(f"提示: 上下文过长（{original_context_length}字符），已截断至{config.max_context_length}字符以避免API超时")

            messages = [
                Message(role="system", content=config.system_prompt),
                Message(role="user", content=f"问题: {query}\n\n上下文:\n{context}")
            ]

            try:
                content = llm_client.generate_response(messages)
                print(f"回答: {content}")
            except Exception as e:
                logging.error(f"LLM请求失败: {e}")
                print(f"抱歉，无法生成答案，请稍后再试。错误信息: {e}")

    except Exception as e:
        print("程序运行时发生严重错误，请检查日志。")


if __name__ == "__main__":
    main()