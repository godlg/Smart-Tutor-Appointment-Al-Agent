"""
可插拔组件层（六类组件 Base / 实现 / Factory）

目标: 让 LLM / Embedding / Splitter / VectorStore / Reranker / Evaluator
六类组件具备统一抽象接口 + YAML 声明式配置 + 工厂懒加载实例化。

约束: 本层是"新增能力", 不重构现有调用路径(knowledge_service /
hybrid_retriever / text_chunker 等保持原样), 通过 ComponentFactory
可选接入, 为后续 EDD 评估铺路。
"""
