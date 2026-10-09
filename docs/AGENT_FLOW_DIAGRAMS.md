# AGENT_FLOW_DIAGRAMS

> P0 阶段产物 #5。 4 个核心 agent 流的时序图,基于 `agents/` 实际代码路径绘制。
>
> 每图都标 **LLM 调用次数** 和 **DB 写次数**,作为后续成本/性能优化基线。

## 1. 任务分类路由 (TaskClassificationAgent)

**触发**: 家长发任何消息到 `/chat/stream` 或 `/chat` 端点
**代码路径**: `api/chat_handler.ProcessUserInput_stream` → `TaskClassificationAgent.classify_task_stream` → `ClassificationProcessor.process_task_stream` → `TaskClassifier.classify` (LLM) → 根据分类结果分发给 `ConsultantAgent` / `TrialAgent` / `LessonAgent` / `UnrelatedHandler`

```mermaid
sequenceDiagram
    autonumber
    participant U as 家长
    participant API as api/chat_handler
    participant TC as TaskClassificationAgent
    participant CL as TaskClassifier (LLM)
    participant SM as StateManager
    participant H as AgentRouter
    participant CA as ConsultantAgent
    participant TA as TrialAgent
    participant LA as LessonAgent
    participant UH as UnrelatedHandler

    U->>API: "初二数学函数基础不好"
    API->>TC: classify_task_stream(text)
    TC->>SM: get_current_state
    SM-->>TC: FlowState.CLASSIFY
    TC->>CL: classify(text) (1 LLM call)
    Note over CL: PromptTemplate:<br/>4 选 1: CONSULT / TRIAL / LESSON / UNRELATED
    CL-->>TC: {"intent": "consultation"}
    TC->>H: route(intent, user_input)
    alt intent == "consultation"
        H->>CA: consult_stream(text)
        CA-->>U: 流式 token 1..N
    else intent == "trial"
        H->>TA: handle_trial(text)
        TA-->>U: 多轮追问或预约成功
    else intent == "lesson"
        H->>LA: handle_lesson(text)
        LA-->>U: 排课建议
    else intent == "unrelated"
        H->>UH: handle_unrelated
        UH-->>U: "我是家教顾问,您是想..."
    end
```

**LLM 调用**: 1 次 (TaskClassifier)
**DB 写**: 0 (但本轮修复后, 后台 `recommendation_service` 会独立写 `user_recommendations`, 跟本流无关)
**典型延迟**: 1-3 秒

---

## 2. 试讲多轮流程 (TrialAgent)

**触发**: `intent == "trial"` 路由过来
**代码路径**: `TrialAgent.__init__.TrialProcessor` → `TrialInputParser.parse_stream` (LLM 提取结构化 JSON) → 多轮补全信息 → `TrialTeacherFinder.find_recommended_teachers` → `TrialProcessor._create_trial` 写库 → 成功消息

```mermaid
sequenceDiagram
    autonumber
    participant U as 家长
    participant TP as TrialProcessor
    participant IP as TrialInputParser (LLM)
    participant TF as TrialTeacherFinder
    participant LS as LessonService
    participant SS as StudentProfileService

    U->>TP: "想约一次数学试讲"
    TP->>IP: parse_stream(text) (1 LLM call)
    Note over IP: 提取 JSON:<br/>{subject, grade_level, start_time,<br/> preferred_teacher_name, info_complete, ...}
    IP-->>TP: JSON
    alt info_complete == false
        TP-->>U: "请问孩子是哪个年级? 什么时候方便?"
        U->>TP: "初三,本周五晚上 7 点"
        TP->>IP: parse_stream(...) (2nd LLM call)
        IP-->>TP: info_complete = true
    end
    TP->>TF: find_recommended_teachers(subject, grade)
    TF->>LS: lesson_repo.check_teacher_conflict (per teacher)
    TF-->>TP: top-3 推荐
    TP-->>U: 列出 3 个老师 + 等待选择
    U->>TP: "选第一个,王老师"
    TP->>IP: parse_stream(confirmation) (3rd LLM call)
    Note over IP: 提取 confirmation
    IP-->>TP: confirmed
    TP->>SS: initialize_default_student
    TP->>LS: create_lesson(teacher, student, type=trial, price=0)
    LS-->>TP: lesson_id
    TP-->>U: 试讲预约成功消息
```

**LLM 调用**: 1-3 次 (每轮追问 + 1 次确认,实际多在 2 次)
**DB 写**: 1 次 lessons + 1 次 user_behaviors (trial_book)
**典型延迟**: 5-15 秒(包含家长输入时间)

---

## 3. RAG 咨询 (ConsultantAgent)

**触发**: `intent == "consultation"` 路由过来
**代码路径**: `ConsultantAgent.consult_stream` → `ConsultationClassifier.is_consultation_related` (LLM) → `KnowledgeRetriever.search` (FAISS) → `ResponseGenerator.generate_stream` (LLM 流式)

```mermaid
sequenceDiagram
    autonumber
    participant U as 家长
    participant CA as ConsultantAgent
    participant CC as ConsultationClassifier (LLM)
    participant KR as KnowledgeRetriever (FAISS)
    participant KS as KnowledgeService
    participant DB as SQLite + FAISS index
    participant RG as ResponseGenerator (LLM, 流式)

    U->>CA: "初二数学函数基础不好怎么办"
    CA->>CC: is_consultation_related(text) (1 LLM call)
    Note over CC: 判别"是不是真正的家教问题"
    alt not related
        CC-->>CA: False
        CA-->>U: "我是家教顾问,您是想..."
    else related
        CC-->>CA: True
        CA->>KR: search("初二数学函数基础", top_k=3)
        KR->>KS: search(query)
        KS->>DB: FAISS index.search + 二次过滤
        DB-->>KS: top-3 文档
        KS-->>KR: docs
        KR-->>CA: docs
        CA->>RG: response_generator.stream(query, docs) (1 LLM call, 流式)
        RG-->>CA: token 1
        CA-->>U: token 1
        RG-->>U: token N
    end
```

**LLM 调用**: 2 次 (判别 + 流式回答,实际流式 1 次)
**DB 读**: 1 次 FAISS 向量搜
**典型延迟**: 2-6 秒(取决于流式返回速度)

---

## 4. 续费/流失提醒派发 (后台调度 + 家长中心)

**触发**: 后台 scheduler 每天 9:00 / 14:00 / 19:00 (或家长主动 POST `/api/user-behavior/send-reminder`)
**代码路径**: `RecommendationService.generate_recommendations_job` → 遍历 students → `recommend_teachers` → 写 `user_recommendations` (限频 20h) → 家长中心 `GET /api/user-behavior/recommendations?mark_sent=true` 拉取并标记

```mermaid
sequenceDiagram
    autonumber
    participant S as Scheduler (schedule lib)
    participant RS as RecommendationService
    participant DB as SQLite
    participant PC as 家长中心 / chat 端

    Note over S: 每天 9/14/19 点
    S->>RS: generate_recommendations_job()
    loop 每个学生
        RS->>DB: user_behavior.get_pending_recommendations(student_<id>)
        alt 20h 内已有 tutor_recommendation
            RS->>RS: 跳过
        else
            RS->>DB: teachers.get_all_teachers()
            loop 每个老师
                RS->>RS: score_teacher_for_student
            end
            RS->>DB: user_behavior.create_recommendation(...)
            Note over DB: insert tutor_recommendation row,<br/>teacher_id, is_sent=0
        end
    end
    Note over PC: 家长下次访问
    PC->>RS: GET /api/user-behavior/recommendations?mark_sent=true
    RS->>DB: get_pending_recommendations(default_user + 所有 student_*)
    RS->>DB: mark_recommendation_sent(id) for each
    RS-->>PC: {count, data: [推荐列表]}
    PC-->>家长: 展示"推荐王老师辅导高中数学"
```

**LLM 调用**: 0 (推荐打分是纯 Python 算法)
**DB 写**: 1-N 次 user_recommendations (限频后通常 0-2)
**DB 读**: 1 次拉取
**典型延迟**: < 1 秒

---

## 5. (P3 规划) 复习提醒流

**未实现, P3 引入**,这里是占位:

```mermaid
sequenceDiagram
    autonumber
    participant S as Scheduler
    participant RD as ReminderDispatcher (P3)
    participant DB as SQLite
    participant PC as 家长中心

    Note over S: 每天 09:00
    S->>RD: scan_review_reminders()
    RD->>DB: student_mastery.scan(mastery<0.5 and last_reviewed>14d)
    DB-->>RD: 知识点列表
    loop 每个 mastery 低的点
        RD->>DB: user_recommendation.create(knowledge_review, ...)
    end
    PC->>RD: GET /api/user-behavior/recommendations
    RD-->>PC: 含 knowledge_review 类型
```

**P3 实现细节见阶段计划 P3**。
