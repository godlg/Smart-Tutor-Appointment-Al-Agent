"""
EDD 评估体系 + 发布门禁 (eval/)

独立模块, 不修改任何现有 Agent / Service 行为 (只 import 调用来评估)。

- cases/   冻结评估集 (routing / rag / slot / tool / handoff / safety, JSON)
- runner.py EDDRunner: 顺序跑六类评估, 量化正确性 + P95 延迟 + Token 成本 + 最大步数
- gates.py  发布门禁: 按 config/evals.yaml 阈值判定 pass/fail
- backfill.py 从线上失败样本回灌评估集 (append + 按 id 去重)

命令行入口 (用 .venv 跑):
    python -m eval.runner                  # 真实跑一轮 + 门禁判定
    python -m eval.runner --dry-run        # 只校验评估集结构 + 编排 (不调 LLM/检索)
    python -m eval.runner --only routing,rag
    python -m eval.backfill --type routing --file failures.json
"""
