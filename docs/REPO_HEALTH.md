# REPO_HEALTH

> 自动生成 by `scripts/audit_repo.py`,运行时间 < 5s。

人手不修改本文件(下次跑 audit 会被覆盖)。


## T1. 代码量分布

| Layer | Files | Lines |
|---|---|---|
| app | 1 | 126 |
| api | 15 | 1929 |
| agents | 34 | 4375 |
| services | 10 | 1576 |
| db | 15 | 2262 |
| web | 2 | 154 |
| tests | 7 | 1052 |
| scripts | 3 | 698 |
| config | 6 | 382 |
| **TOTAL** | **93** | **12554** |

## T2. 分层合规

| File | Import | Reason |
|---|---|---|
| services\lesson_service.py | agents.lesson | services 不允许 import agents 层 |

## T3. 死代码候选

| File | Public Name |
|---|---|
| _syntax_check.py | errors |
| _syntax_check.py | fail |
| _syntax_check.py | files |
| _syntax_check.py | ok |
| _syntax_check.py | project_root |
| .github\skills\setup-environment\scripts\verify_env.py | CHAT_PROVIDERS |
| .github\skills\setup-environment\scripts\verify_env.py | EMBEDDING_PROVIDERS |
| .github\skills\setup-environment\scripts\verify_env.py | PACKAGES |
| .github\skills\setup-environment\scripts\verify_env.py | PLACEHOLDER_HINTS |
| .github\skills\setup-environment\scripts\verify_env.py | check_env |
| .github\skills\setup-environment\scripts\verify_env.py | check_imports |
| .github\skills\setup-environment\scripts\verify_env.py | main |
| .github\skills\skill-creator\scripts\init_skill.py | EXAMPLE_ASSET |
| .github\skills\skill-creator\scripts\init_skill.py | EXAMPLE_REFERENCE |
| .github\skills\skill-creator\scripts\init_skill.py | EXAMPLE_SCRIPT |
| .github\skills\skill-creator\scripts\init_skill.py | SKILL_TEMPLATE |
| .github\skills\skill-creator\scripts\init_skill.py | init_skill |
| .github\skills\skill-creator\scripts\init_skill.py | main |
| .github\skills\skill-creator\scripts\init_skill.py | title_case_skill_name |
| .github\skills\skill-creator\scripts\package_skill.py | main |
| .github\skills\skill-creator\scripts\package_skill.py | package_skill |
| .github\skills\skill-creator\scripts\quick_validate.py | validate_skill |
| agents\consultant\consultation_classifier.py | ConsultationClassifier |
| agents\consultant\consultation_processor.py | ConsultationProcessor |
| agents\consultant\knowledge_retriever.py | KnowledgeRetriever |
| agents\consultant\prompt_builder.py | PromptBuilder |
| agents\consultant\response_generator.py | ResponseGenerator |
| agents\lesson\lesson_agent.py | LessonAgent |
| agents\lesson\lesson_processor.py | LessonInputParser |
| agents\lesson\lesson_processor.py | LessonMessageBuilder |
| agents\lesson\lesson_processor.py | LessonProcessor |
| agents\lesson\lesson_processor.py | LessonTeacherFinder |
| agents\lesson\reschedule_handler.py | RescheduleHandler |
| agents\lesson\reschedule_handler.py | RescheduleMode |
| agents\lesson\reschedule_handler.py | logger |
| agents\task_classification\agent_router.py | AgentRouter |
| agents\task_classification\classification_processor.py | ClassificationProcessor |
| agents\task_classification\state_manager.py | FlowState |
| agents\task_classification\state_manager.py | SharedState |
| agents\task_classification\state_manager.py | StateManager |

(共 217 个,只列前 40)


注: `__init__.py` 里的 public 名(用于 re-export)和 `app.py` 顶层 `app` 已被排除。

已知限制: 本检查不跟踪 `from pkg import *` 和 `__all__` re-export 链,例如

`agents/consultant/{ConsultationClassifier, ConsultationProcessor, ...}`

虽然出现在 `agents/consultant/__init__.py` 的 `__all__` 里被 re-export,

但本表仍会显示为候选(实际被 `ConsultantAgent` 使用)。 复核请查 `__init__.py`。


## T4. 未使用 import (本仓库内)

| File | Import |
|---|---|
| api\lesson.py | services.teacher_service.TeacherService |
| db\__init__.py | db_router.DatabaseRouter |
| scripts\migrate_schema.py | db.base.schema_migrator._get_columns |
| tests\test_conversion_logic.py | agents.trial.conversion_logic.ConversionStrategy |
| tests\test_schedule_generator.py | agents.lesson.conflict_detector.Conflict |
| tests\test_schedule_generator.py | agents.lesson.conflict_detector.ConflictType |
| tests\test_schedule_generator.py | agents.lesson.conflict_detector.ConflictSeverity |

## T5. 外网 URL 依赖 (源码内,排除注释/docstring)

| File | URL |
|---|---|
| scripts\demo_e2e.py | http://127.0.0.1:8001 |

## 已知警告 (非 P0 范围)

- `db/models.py:18` SQLAlchemy 2.0 `declarative_base` 已迁,但 `datetime.utcnow()` 仍有 ~3 处弃用警告。

- `agents/consultant/__init__.py` 注释与 README 早期版本不一致,本表不查语义。
