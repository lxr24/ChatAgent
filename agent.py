"""
主入口文件 - 启动RAG Agent
"""
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

            # 查询向量存储
            query_embedding = embedding_model.embed_query(query)
            results = vector_store.search(query_embedding, top_k=config.vector_store.top_k)

            # 构建上下文
            context = "\n\n".join([f"来源: {chunk.metadata['source']}\n内容: {chunk.content}" for chunk, _ in results])

            messages = [
                Message(role="system", content=config.system_prompt),
                Message(role="user", content=f"问题: {query}\n\n上下文:\n{context}")
            ]

            try:
                content = llm_client.generate_response(messages)
                print(f"回答:INTACT {content}")
            except Exception as e:
                print(f"抱歉，无法生成答案，请稍后再试。错误信息: {e}")

    except Exception as e:
        print("程序运行时发生严重错误，请检查日志。")


if __name__ == "__main__":
    main()