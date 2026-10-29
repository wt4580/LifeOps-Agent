import json
from datetime import datetime
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.src.common.config.db_config import Base
from app.src.common.config.knowledge_db_config import KnowledgeBase
from app.src.domain.entity.chat_entity import ChatMessage
from app.src.service import evidence_service
from app.src.util.retrieval import retrieval


@pytest.fixture
def evidence_db(tmp_path, monkeypatch):
    business = create_engine(f"sqlite:///{tmp_path / 'business.db'}")
    knowledge = create_engine(f"sqlite:///{tmp_path / 'knowledge.db'}")
    Base.metadata.create_all(business)
    KnowledgeBase.metadata.create_all(knowledge)
    factory = sessionmaker(bind=business, expire_on_commit=False)
    monkeypatch.setattr(evidence_service, "SessionLocal", factory)
    monkeypatch.setattr(retrieval, "KnowledgeSessionLocal", sessionmaker(bind=knowledge))
    yield factory
    business.dispose()
    knowledge.dispose()


def test_statements_are_identity_scoped_bounded_and_keep_original_time(evidence_db):
    with evidence_db() as db:
        for i in range(23):
            db.add(ChatMessage(session_id="alice", role="user", content=f"statement-{i}", created_at=datetime(2020, 1, 1)))
        db.add(ChatMessage(session_id="alice", role="assistant", content="你今天吃了火锅"))
        db.add(ChatMessage(session_id="bob", role="user", content="Bob ate pizza"))
        current = ChatMessage(session_id="alice", role="user", content="我今天吃得合适吗")
        db.add(current)
        db.flush()
        current_id = current.id
        db.add(ChatMessage(session_id="alice", role="user", content="future input"))
        db.commit()
    statements = evidence_service.load_user_statements("alice", before_message_id=current_id, limit=9999)
    assert len(statements) == 20
    assert statements[0]["content"] == "statement-3"
    assert statements[-1]["content"] == "statement-22"
    assert all(row["recorded_at_utc"] == "2020-01-01T00:00:00Z" for row in statements)
    assert all(row["evidence_role"] == "user_statement" for row in statements)
    assert evidence_service.load_user_statements("") == []


def test_rag_separates_reference_first_person_text_from_actual_user_statements(evidence_db, monkeypatch):
    completion = Mock(return_value="根据你说的早餐，再参考指南评价。[1]")
    monkeypatch.setattr(retrieval, "chat_completion", completion)
    citations = [retrieval.Citation(path="other_person_diary.txt", page=None, score=60,
                                   doc_topic="personal_log", snippet="我今天吃了蛋糕。这是第三方资料中的第一人称示例。")]
    retrieval.rag_answer("我早餐吃了包子，合适吗", citations, session_id="alice",
                         user_statements=[{"content": "我早餐吃了包子", "recorded_at_utc": "2026-10-29T00:00:00Z"}],
                         tool_results=[{"action": "query_database", "result": "已记录的用户原话"}])
    messages = completion.call_args.args[0]
    assert "第一人称不能赋予个人事实身份" in messages[0]["content"]
    assert "相对于 recorded_at_utc" in messages[0]["content"]
    assert "reference_material" in messages[1]["content"]
    personal = json.loads(messages[1]["content"].split("PersonalContext:\n")[1].split("\n\nContext:\n")[0])
    assert personal["recent_user_statements"][0]["content"] == "我早餐吃了包子"
    assert "蛋糕" not in json.dumps(personal, ensure_ascii=False)
    assert personal["tool_results"][0]["action"] == "query_database"


def test_personal_history_has_a_total_prompt_budget(evidence_db):
    with evidence_db() as db:
        db.add_all([ChatMessage(session_id="alice", role="user", content="x" * 1000) for _ in range(20)])
        db.commit()
    statements = evidence_service.load_user_statements("alice")
    assert sum(len(item["content"]) for item in statements) == 6000
    assert len(statements) == 6


def test_reference_prompt_is_bounded_without_hiding_the_matched_chunk(evidence_db, monkeypatch):
    completion = Mock(return_value="bounded")
    monkeypatch.setattr(retrieval, "chat_completion", completion)
    citations = [retrieval.Citation(f"guide-{i}.txt", None, "材料" * 10000, 60) for i in range(20)]
    retrieval.rag_answer("参考资料", citations)
    content = completion.call_args.args[0][1]["content"]
    assert len(content) < 13000
    assert "参考内容已截断" in content
    from app.src.domain.entity.knowledge_entity import DocumentChunk
    with retrieval.KnowledgeSessionLocal() as db:
        rows = [DocumentChunk(file_path="neighbors.txt", chunk_text="before" * 500, chunk_hash=str(i)) for i in range(2)]
        matched = DocumentChunk(file_path="neighbors.txt", chunk_text="MATCHED_EVIDENCE_" + "x" * 3000, chunk_hash="hit")
        db.add_all([*rows, matched])
        db.flush()
        chunk_id = matched.id
        db.commit()
    retrieval.rag_answer("参考资料", [retrieval.Citation("neighbors.txt", None, "fallback", 60, chunk_id=chunk_id)])
    content = completion.call_args.args[0][1]["content"]
    assert "MATCHED_EVIDENCE_" in content and "beforebefore" not in content


def test_missing_personal_history_is_explicit_not_inferred_from_guidelines(evidence_db, monkeypatch):
    completion = Mock(return_value="不知道你吃了什么，请补充。")
    monkeypatch.setattr(retrieval, "chat_completion", completion)
    retrieval.rag_answer("我今天吃得怎么样", [retrieval.Citation("guide.txt", None, "我今天吃了燕麦是参考示例，不属于用户。", 60)])
    content = completion.call_args.args[0][1]["content"]
    personal = json.loads(content.split("PersonalContext:\n")[1].split("\n\nContext:\n")[0])
    assert personal["recent_user_statements"] == []
    assert "并询问所缺的信息" in completion.call_args.args[0][0]["content"]


def test_graph_passes_scoped_history_and_prior_read_results_to_rag(evidence_db, monkeypatch):
    from app.src.agent import graph_chat as graph
    with evidence_db() as db:
        db.add(ChatMessage(session_id="alice", role="user", content="我早餐吃了包子"))
        db.add(ChatMessage(session_id="bob", role="user", content="Bob pizza"))
        current = ChatMessage(session_id="alice", role="user", content="我早餐合适吗")
        db.add(current)
        db.flush()
        current_id = current.id
        db.commit()
    monkeypatch.setattr(graph, "search_chunks", lambda *a, **k: [retrieval.Citation("guide", None, "包子早餐评价指南", 60)])
    answer = Mock(return_value="已按来源区分")
    monkeypatch.setattr(graph, "rag_answer", answer)
    state = {"session_id": "alice", "operation_id": f"message:{current_id}", "user_message": "我早餐合适吗",
             "decision": {"action": "query_knowledge", "args": {}}, "plan_index": 2,
             "plan": [{"action": "query_database", "result": "历史用户陈述"},
                      {"action": "record_expense", "result": "不应作为饮食自述"},
                      {"action": "query_accounts", "result": "未来步骤不能作为已查询"}]}
    result = graph._node_run_tool(state)
    assert [item["content"] for item in answer.call_args.kwargs["user_statements"]] == ["我早餐吃了包子"]
    assert answer.call_args.kwargs["tool_results"] == [{"action": "query_database", "result": "历史用户陈述"}]
    assert result["citations"][0]["evidence_role"] == "reference_material"


def test_nonpersonal_reference_query_does_not_load_personal_history(evidence_db, monkeypatch):
    from app.src.agent import graph_chat as graph
    loader = Mock(side_effect=AssertionError("Unnecessary personal history query"))
    monkeypatch.setattr(evidence_service, "load_user_statements", loader)
    monkeypatch.setattr(graph, "search_chunks", lambda *a, **k: [retrieval.Citation("manual", None, "报销材料", 60)])
    monkeypatch.setattr(graph, "rag_answer", Mock(return_value="报销材料"))
    graph._node_run_tool({"session_id": "alice", "user_message": "报销需要哪些材料",
                          "decision": {"action": "query_knowledge", "args": {}}})
    loader.assert_not_called()
