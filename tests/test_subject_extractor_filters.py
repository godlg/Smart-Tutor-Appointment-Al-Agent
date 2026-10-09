"""P0-1 / P0-2:学科-年级抽取 + filter-then-vector 透传 + SQL NULL 豁免 单元测试。

覆盖四组:
A. 抽取器(纯规则):单条消息抽学科/年级、年级词映射、无命中返回 None、跨学科词不误抽
B. SQL NULL 豁免:subject / grade_level 为 NULL 的跨学科(跨年级)文档不被误杀,
   doc_type 仍严格等值
C. 聊天链路透传:dense / hybrid 两条路都收到 subject / grade_level;
   process_consultation_stream 检索前抽好并透传,输出不含抽取错误
D. 向后兼容:不带过滤参数时底层显式收到 subject=None / grade_level=None(全量检索)

不依赖远程 embedding:需要向量的用例统一注入确定性 fake embedder。
不硬编码日期。
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run(coro):
    return asyncio.run(coro)


def _fake_embed(text: str) -> List[float]:
    """sha256 → 16 维确定性向量(不依赖远程 API)。"""
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:16]]


@pytest.fixture(autouse=True)
def _patch_embed(monkeypatch):
    """全局替换 embed_input: text_embedding / knowledge_service / hybrid_retriever 三处。"""
    from services import text_embedding, knowledge_service, hybrid_retriever

    monkeypatch.setattr(text_embedding, "embed_input", _fake_embed)
    monkeypatch.setattr(knowledge_service, "embed_input", _fake_embed)
    monkeypatch.setattr(hybrid_retriever, "embed_input", _fake_embed)


@pytest.fixture
def in_memory_db():
    """全新内存 SQLite(schema + migration 已跑过)。"""
    from db.db_router import DatabaseRouter

    router = DatabaseRouter("sqlite:///:memory:")
    yield router
    router.close()


def _make_history(*texts):
    """把若干文本按「用户/顾问」交替塞进 InMemoryChatMessageHistory。"""
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.messages import AIMessage, HumanMessage

    history = InMemoryChatMessageHistory()
    for i, text in enumerate(texts):
        history.add_message(HumanMessage(content=text) if i % 2 == 0 else AIMessage(content=text))
    return history


# ======================================================================
# A. 抽取器(纯规则)
# ======================================================================

class TestExtractSubject:
    def test_extracts_subject_from_single_message(self):
        from agents.consultant.subject_extractor import extract_subject

        assert extract_subject("高一数学想提分,有什么课程?") == "数学"
        assert extract_subject("孩子初二,英语跟不上") == "英语"

    def test_every_whitelist_subject_is_recognized(self):
        from agents.consultant.subject_extractor import SUBJECT_WHITELIST, extract_subject

        for subject in SUBJECT_WHITELIST:
            assert extract_subject(f"我想补{subject}") == subject

    def test_single_char_shorthand_not_matched(self):
        """「数」这类单字不算命中(白名单只收完整学科名)。"""
        from agents.consultant.subject_extractor import extract_subject

        assert extract_subject("数一数二") is None

    def test_no_hit_returns_none(self):
        from agents.consultant.subject_extractor import extract_subject

        assert extract_subject("孩子上课注意力不集中怎么办") is None
        assert extract_subject("") is None
        assert extract_subject(None) is None

    @pytest.mark.parametrize(
        "text",
        ["跨学科辅导怎么安排", "有没有全科套餐", "学科匹配怎么看", "理综怎么复习", "文综怎么提分"],
    )
    def test_cross_subject_words_not_extracted(self, text):
        """跨学科表述不是具体学科,不能被误抽(否则会把硬约束锁死在错误学科上)。"""
        from agents.consultant.subject_extractor import extract_subject

        assert extract_subject(text) is None


class TestExtractGradeLevel:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("初二学生数学不好", "初中"),
            ("初三冲刺", "初中"),
            ("七年级", "初中"),
            ("高一数学想提分", "高中"),
            ("高二物理", "高中"),
            ("高三冲刺套餐", "高中"),
            ("六年级阅读", "小学"),
            ("小学阅读启蒙课", "小学"),
            # 直接命中年级段本身也认
            ("小学阶段怎么学", "小学"),
            ("初中数学 一元二次方程", "初中"),
            ("高中物理 力学", "高中"),
        ],
    )
    def test_grade_word_mapping(self, text, expected):
        from agents.consultant.subject_extractor import extract_grade_level

        assert extract_grade_level(text) == expected

    def test_no_hit_returns_none(self):
        from agents.consultant.subject_extractor import extract_grade_level

        assert extract_grade_level("孩子成绩怎么提高") is None
        assert extract_grade_level("") is None
        assert extract_grade_level(None) is None

    def test_extract_subject_grade_returns_pair(self):
        from agents.consultant.subject_extractor import extract_subject_grade

        assert extract_subject_grade("孩子高一,数学想提分") == ("数学", "高中")
        assert extract_subject_grade("随便聊聊") == (None, None)


class TestResolveSubjectGrade:
    def test_query_wins_over_history(self):
        """本轮明确说出的学科/年级优先于历史残留。"""
        from agents.consultant.subject_extractor import resolve_subject_grade

        history = _make_history("我家孩子初二,数学不好", "好的,初二数学我来答疑")
        assert resolve_subject_grade("高一物理怎么学", history=history) == ("物理", "高中")

    def test_history_fills_the_missing_field(self):
        """本轮只说了学科 → 年级从最近消息里补。"""
        from agents.consultant.subject_extractor import resolve_subject_grade

        history = _make_history("我家孩子高一,数学不好", "收到")
        assert resolve_subject_grade("那物理呢?", history=history) == ("物理", "高中")

    def test_only_recent_messages_are_scanned(self):
        """更早历史(超出最近 N 条)里的学科不应污染当前查询。"""
        from agents.consultant.subject_extractor import (
            RECENT_MESSAGE_LIMIT,
            resolve_subject_grade,
        )

        # 前 2 条带学科,后面再补满 RECENT_MESSAGE_LIMIT 条 → 带学科的两条被挤出窗口
        older = ["小学语文怎么教", "收到"]
        filler = ["嗯"] * RECENT_MESSAGE_LIMIT
        history = _make_history(*(older + filler))
        assert resolve_subject_grade("有什么课程", history=history) == (None, None)

    def test_no_hit_returns_none(self):
        from agents.consultant.subject_extractor import resolve_subject_grade

        assert resolve_subject_grade("怎么提高注意力") == (None, None)

    def test_no_history_placeholder_is_not_a_subject(self):
        """「暂无历史消息」是占位文案,不能被抽成学科「历史」。"""
        from agents.consultant.subject_extractor import resolve_subject_grade

        assert resolve_subject_grade("有什么课程", context_texts=("暂无历史消息",)) == (None, None)

    def test_context_texts_are_used_after_query(self):
        from agents.consultant.subject_extractor import resolve_subject_grade

        subject, grade = resolve_subject_grade(
            "计算薄弱", context_texts=("用户: 初二,数学不行",)
        )
        assert (subject, grade) == ("数学", "初中")

    def test_profile_provider_only_used_as_last_resort(self):
        from agents.consultant.subject_extractor import resolve_subject_grade

        provider = lambda: ("数学", "高中")  # noqa: E731
        assert resolve_subject_grade("有什么课程", profile_provider=provider) == ("数学", "高中")
        # 已抽到 → 不触发兜底
        assert resolve_subject_grade("高一物理", profile_provider=provider) == ("物理", "高中")

    def test_profile_provider_failure_returns_none(self):
        from agents.consultant.subject_extractor import resolve_subject_grade

        def _boom():
            raise RuntimeError("画像库不可用")

        assert resolve_subject_grade("有什么课程", profile_provider=_boom) == (None, None)


class TestDefaultProfileFilters:
    """优先级 4 的兜底实现(默认学生画像)。"""

    def _patch_service(self, monkeypatch, students):
        import services.student_profile_service as sps

        fake = MagicMock()
        fake.get_or_create_default_parent.return_value = {"id": 1}
        fake.get_students_by_parent.return_value = students
        monkeypatch.setattr(sps, "StudentProfileService", lambda: fake)

    def test_reads_weak_subject_and_grade(self, monkeypatch):
        from agents.consultant.subject_extractor import default_profile_filters

        self._patch_service(
            monkeypatch, [{"grade_level": "高中", "weak_subjects": ["数学", "物理"]}]
        )
        assert default_profile_filters() == ("数学", "高中")

    def test_specific_grade_word_is_normalized(self, monkeypatch):
        from agents.consultant.subject_extractor import default_profile_filters

        self._patch_service(monkeypatch, [{"grade_level": "高一", "weak_subjects": ["物理"]}])
        assert default_profile_filters() == ("物理", "高中")

    def test_no_students_returns_none(self, monkeypatch):
        from agents.consultant.subject_extractor import default_profile_filters

        self._patch_service(monkeypatch, [])
        assert default_profile_filters() == (None, None)

    def test_service_failure_returns_none(self, monkeypatch):
        import services.student_profile_service as sps

        def _boom():
            raise RuntimeError("DB 不可用")

        monkeypatch.setattr(sps, "StudentProfileService", _boom)
        from agents.consultant.subject_extractor import default_profile_filters

        assert default_profile_filters() == (None, None)


# ======================================================================
# B. SQL NULL 豁免(P0-2)
# ======================================================================

def _seed_repo(repo):
    """4 条:具体学科两条 + 跨学科(None/高中) + 跨学科跨年级(None/None)。"""
    repo.add_document(
        content="高中数学 12 周提分方案:函数、几何、数列、概率四大模块。",
        category="course_intro", keywords=["高中", "数学"],
        embedding=_fake_embed("高中数学"),
        subject="数学", grade_level="高中", doc_type="course_intro",
    )
    repo.add_document(
        content="高三冲刺套餐:针对高考设计,适合数学、英语、理综等全学科。",
        category="course_intro", keywords=["高三", "冲刺"],
        embedding=_fake_embed("高三冲刺"),
        subject=None, grade_level="高中", doc_type="course_intro",
    )
    repo.add_document(
        content="小学阅读启蒙课:以绘本为载体,建立自主阅读习惯。",
        category="course_intro", keywords=["小学", "阅读"],
        embedding=_fake_embed("小学阅读"),
        subject="语文", grade_level="小学", doc_type="course_intro",
    )
    repo.add_document(
        content="如何帮孩子选择合适的家教老师:先看学科匹配,再看年级段。",
        category="parent_guide", keywords=["家长", "老师"],
        embedding=_fake_embed("家长指南"),
        subject=None, grade_level=None, doc_type="parent_guide",
    )


class TestSqlNullExemption:
    def test_subject_filter_keeps_cross_subject_docs(self, in_memory_db):
        repo = in_memory_db.knowledge_repo
        _seed_repo(repo)

        results = repo.search_documents_with_filters(subject="数学")
        subjects = [r["subject"] for r in results]

        assert "数学" in subjects
        assert None in subjects, "subject=None 的跨学科文档不应被学科过滤误杀"
        assert "语文" not in subjects, "其它学科仍必须被挡掉"

    def test_grade_filter_keeps_cross_grade_docs(self, in_memory_db):
        repo = in_memory_db.knowledge_repo
        _seed_repo(repo)

        results = repo.search_documents_with_filters(grade_level="高中")
        grades = [r["grade_level"] for r in results]

        assert "高中" in grades
        assert None in grades, "grade_level=None 的跨年级文档不应被年级过滤误杀"
        assert "小学" not in grades, "其它年级段仍必须被挡掉"

    def test_combined_filter_keeps_matching_and_null_only(self, in_memory_db):
        repo = in_memory_db.knowledge_repo
        _seed_repo(repo)

        results = repo.search_documents_with_filters(subject="数学", grade_level="高中")
        contents = " ".join(r["content"] for r in results)

        assert "高中数学" in contents
        assert "高三冲刺" in contents, "subject=None 的跨学科文档应保留"
        assert "如何帮孩子选择" in contents, "subject/grade 都 None 的文档应保留"
        assert "小学阅读" not in contents

    def test_doc_type_stays_strict_equality(self, in_memory_db):
        """doc_type 无豁免语义:NULL 不会被 doc_type 过滤放行。"""
        repo = in_memory_db.knowledge_repo
        _seed_repo(repo)

        results = repo.search_documents_with_filters(doc_type="parent_guide")
        assert [r["content"] for r in results] == [
            "如何帮孩子选择合适的家教老师:先看学科匹配,再看年级段。"
        ]

    def test_no_filter_returns_all(self, in_memory_db):
        repo = in_memory_db.knowledge_repo
        _seed_repo(repo)

        assert len(repo.search_documents_with_filters()) == 4


class TestServiceFilterThenVectorNullExemption:
    """service 层 filter-then-vector:硬约束生效,同时不误杀跨学科文档。"""

    def _build_service(self, router):
        from services.knowledge_service import KnowledgeService

        svc = KnowledgeService(db_path="sqlite:///:memory:")
        svc.db_router = router
        svc.db = router.knowledge_repo
        svc.document_ids = []
        svc.index = None
        svc._index_dirty = False
        svc.initialized = False
        return svc

    def test_filtered_search_keeps_cross_subject_and_blocks_other_subject(self, in_memory_db):
        _seed_repo(in_memory_db.knowledge_repo)
        svc = self._build_service(in_memory_db)
        _run(svc.initialize())

        results = _run(svc.search(
            query="高一数学 提分 课程", top_k=5, subject="数学", grade_level="高中",
        ))
        contents = [r["content"] for r in results]

        assert contents, "过滤后不应是空结果"
        assert any("高中数学" in c for c in contents)
        assert any("高三冲刺" in c for c in contents), "跨学科文档应仍可召回"
        assert not any("小学阅读" in c for c in contents), "小学语文必须被硬约束挡掉"


# ======================================================================
# C / D. 聊天链路透传 + 向后兼容
# ======================================================================

def _make_retriever(mock_search, mode: str = "dense"):
    """构造 KnowledgeRetriever,用 mock 顶替底层检索(隔离真实检索 / LLM)。"""
    from agents.consultant.knowledge_retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever()
    retriever.retrieval_mode = mode
    service = MagicMock()
    service.search = mock_search
    retriever.knowledge_service = service
    retriever.kb_initialized = True  # 跳过真实初始化
    return retriever


def _make_hybrid_retriever(mock_search):
    from agents.consultant.knowledge_retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever()
    retriever.retrieval_mode = "hybrid"
    retriever.kb_initialized = True
    retriever.hybrid_retriever = MagicMock()
    retriever.hybrid_retriever.search = mock_search
    return retriever


class TestSearchKnowledgeFilterPassThrough:
    def test_dense_path_receives_subject_and_grade(self):
        mock = AsyncMock(return_value=[{"content": "doc"}])
        retriever = _make_retriever(mock, "dense")

        _run(retriever.search_knowledge(
            "高一数学怎么提分", top_k=3, subject="数学", grade_level="高中"
        ))

        mock.assert_awaited_once_with(
            "高一数学怎么提分", top_k=3, subject="数学", grade_level="高中"
        )

    def test_hybrid_path_receives_subject_and_grade(self):
        mock = AsyncMock(return_value=[{"content": "doc"}])
        retriever = _make_hybrid_retriever(mock)

        _run(retriever.search_knowledge(
            "高一数学怎么提分", top_k=3, subject="数学", grade_level="高中"
        ))

        mock.assert_awaited_once_with(
            "高一数学怎么提分", top_k=3, subject="数学", grade_level="高中"
        )

    def test_backward_compatible_without_filters(self):
        """D: 不带过滤参数时仍走全量 —— 底层显式收到 subject=None / grade_level=None。"""
        mock = AsyncMock(return_value=[{"content": "doc"}])
        retriever = _make_retriever(mock, "dense")

        _run(retriever.search_knowledge("计算薄弱", top_k=3))

        mock.assert_awaited_once_with("计算薄弱", top_k=3, subject=None, grade_level=None)

    def test_query_context_still_spliced_with_filters(self):
        """过滤参数不影响原有 query_context 拼接行为。"""
        mock = AsyncMock(return_value=[])
        retriever = _make_retriever(mock, "dense")

        _run(retriever.search_knowledge(
            "计算薄弱", top_k=3, query_context="用户: 初二,数学不行",
            subject="数学", grade_level="初中",
        ))

        mock.assert_awaited_once_with(
            "用户: 初二,数学不行\n计算薄弱", top_k=3,
            subject="数学", grade_level="初中",
        )


class TestProcessorExtractionPassThrough:
    """process_consultation_stream:检索前抽学科/年级并透传,输出不含抽取错误。"""

    def _make_processor(self, retriever, response_text: str = "回答"):
        from agents.consultant.consultation_processor import ConsultationProcessor

        generator = MagicMock()

        async def _stream(_user_input, _docs):
            for token in response_text:
                yield token

        generator.generate_response_stream = _stream

        processor = ConsultationProcessor(retriever, MagicMock(), generator)
        # 屏蔽用户行为记录,保持测试聚焦且不触碰存储
        processor._record_consultation_behavior = AsyncMock()
        return processor

    async def _collect(self, async_gen):
        return [token async for token in async_gen]

    def test_stream_extracts_from_query_and_passes_down(self):
        mock = AsyncMock(return_value=[{"content": "高中数学 12 周提分方案"}])
        processor = self._make_processor(_make_retriever(mock, "dense"))

        out = _run(self._collect(processor.process_consultation_stream(
            "孩子高一,数学想提分,有什么课程?", "sess-p01"
        )))
        text = "".join(out)

        assert text == "回答"
        assert "抱歉" not in text and "错误" not in text, "抽取链路不应产生任何错误输出"
        mock.assert_awaited_once_with(
            "孩子高一,数学想提分,有什么课程?", top_k=3,
            subject="数学", grade_level="高中",
        )

    def test_stream_uses_session_history_when_query_has_no_grade(self):
        mock = AsyncMock(return_value=[{"content": "doc"}])
        processor = self._make_processor(_make_retriever(mock, "dense"))
        history = _make_history("我家孩子高一,数学不好", "收到")

        _run(self._collect(processor.process_consultation_stream(
            "那物理呢?", "sess-p02", history=history
        )))

        mock.assert_awaited_once_with(
            "那物理呢?", top_k=3, subject="物理", grade_level="高中"
        )

    def test_stream_uses_profile_provider_as_last_resort(self):
        mock = AsyncMock(return_value=[])
        processor = self._make_processor(_make_retriever(mock, "dense"))
        processor.profile_provider = lambda: ("数学", "高中")

        _run(self._collect(processor.process_consultation_stream(
            "有什么课程推荐?", "sess-p03"
        )))

        mock.assert_awaited_once_with(
            "有什么课程推荐?", top_k=3, subject="数学", grade_level="高中"
        )

    def test_stream_without_extractable_context_stays_unfiltered(self):
        """D: 抽不到学科/年级且无兜底 → 保持全量检索(向后兼容)。"""
        mock = AsyncMock(return_value=[])
        processor = self._make_processor(_make_retriever(mock, "dense"))

        _run(self._collect(processor.process_consultation_stream(
            "孩子上课注意力不集中怎么办", "sess-p04"
        )))

        mock.assert_awaited_once_with(
            "孩子上课注意力不集中怎么办", top_k=3, subject=None, grade_level=None
        )

    def test_stream_passes_filters_on_hybrid_path(self):
        mock = AsyncMock(return_value=[{"content": "doc"}])
        processor = self._make_processor(_make_hybrid_retriever(mock))

        _run(self._collect(processor.process_consultation_stream(
            "高二物理电磁学怎么补", "sess-p05"
        )))

        mock.assert_awaited_once_with(
            "高二物理电磁学怎么补", top_k=3, subject="物理", grade_level="高中"
        )
