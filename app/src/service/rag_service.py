"""RAG服务层 - 封装知识库检索相关的业务逻辑"""

from __future__ import annotations

import logging

from ..common.config.base_config import settings
from ..util.retrieval.retrieval import index_documents, rag_answer, search_chunks

logger = logging.getLogger(__name__)


class RAGService:
    """RAG（检索增强生成）服务
    
    职责：
    1. 索引文档
    2. 基于知识库回答问题
    3. 搜索相关片段
    """
    
    def index_docs(self) -> dict:
        """索引指定目录下的文档
        
        Args:
            docs_dir: 文档目录路径，如果为None则使用配置中的默认值
            
        Returns:
            索引结果统计
        """
        result = index_documents()
        logger.info("Indexing completed: %s", result)
        
        return result
    
    def ask_question(self, question: str) -> dict:
        """基于知识库回答问题
        
        Args:
            question: 用户问题
            
        Returns:
            包含答案和引用的字典
        """
        logger.info("Processing RAG question: %s", question[:100])

        citations = search_chunks(question, top_k=settings.rag_top_k)
        answer_text = rag_answer(question, citations)

        return {
            "answer": answer_text,
            "citations": [
                {
                    "path": c.path,
                    "page": c.page,
                    "snippet": c.snippet,
                    "score": c.score,
                    "reason": c.reason,
                    "source_type": c.source_type,
                    "doc_topic": c.doc_topic,
                }
                for c in citations
            ],
        }
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """搜索知识库中的相关片段
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            
        Returns:
            相关片段列表
        """
        logger.info("Searching knowledge base for: %s", query[:100])
        
        chunks = search_chunks(query, top_k=top_k)
        
        return [
            {
                "path": chunk.path,
                "page": chunk.page,
                "snippet": chunk.snippet,
                "score": chunk.score,
                "reason": chunk.reason,
                "source_type": chunk.source_type,
                "doc_topic": chunk.doc_topic,
            }
            for chunk in chunks
        ]


# 单例模式
rag_service = RAGService()
