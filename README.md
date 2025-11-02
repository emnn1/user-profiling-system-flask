# 用户画像系统（User Profiling System）

## 项目概述

本项目是一套用户画像原型系统，融合了可解释的业务规则与异构图神经网络（HGT）嵌入。后端基于 Flask，提供画像查询、推荐、图刷新、模型训练与运维控制等 RESTful API；前端使用 Streamlit 构建多页面控制台，方便运营与数据团队实时观测与干预。

## 核心能力

- **混合画像评分**：规则引擎与 HGT 嵌入按门控方式融合，兼顾解释性与表现力。
- **增量数据摄取**：模拟实时事件持续写入缓存，为增量学习与推荐输出提供数据源。
- **全量图刷新流程**：一条指令触发图重建、模型训练与融合核心调优，并同步更新嵌入缓存。
- **可解释性分析**：基于 SHAP 的解释服务展示规则贡献、特征权重与最终得分组成。
- **运维可视化**：Streamlit 面板集中展示健康状态、运行指标，并支持后台任务的手动控制。

## 目录结构

| 路径 | 职责说明 |
| --- | --- |
| `backend/app/main.py` | Flask 入口：设备选择、服务实例化、初始训练与路由注册。 |
| `backend/app/api/profiling.py` | 画像相关 API 蓝图：画像、推荐、规则管理、图刷新及运维接口。 |
| `backend/app/core/async_runner.py` | 在线程中托管 asyncio 事件循环，使 Flask 环境也可调度异步任务。 |
| `backend/app/data_source/mock_data_provider.py` | 生成/持久化模拟数据，提供异步实时事件流。 |
| `backend/app/graph_services/graph_builder.py` | 构建 PyG `HeteroData`、维护节点映射、支持事件驱动的增量更新。 |
| `backend/app/ml_models/feature_store.py` | 依据节点属性构建多模态特征编码。 |
| `backend/app/ml_models/hgt_model.py` | HGT 模型结构与对比学习损失实现。 |
| `backend/app/ml_models/encoders.py` | 数值/类别特征编码器组件。 |
| `backend/app/services/data_ingestion.py` | 负责用户画像缓存、行为统计及事件队列管理。 |
| `backend/app/services/incremental_learner.py` | 处理事件增量、刷新用户嵌入（实现位于文件中）。 |
| `backend/app/services/hybrid_profiling_service.py` | 组合规则结果与嵌入向量，产出最终画像分并支持融合核心训练。 |
| `backend/app/services/explainer.py` | SHAP 解释服务，返回贡献明细与 gate 参数。 |
| `backend/app/services/model_trainer.py` | HGT 训练核心：遮蔽拆分、自动化训练与评估。 |
| `backend/app/services/refresh_orchestrator.py` | 调度图刷新、训练、缓存更新的统一编排器。 |
| `backend/app/services/system_controller.py` | 统一运维入口：管理摄取、增量循环、训练任务与指标。 |
| `frontend/app.py` | Streamlit 总览页：健康检查、全量刷新表单与指标可视化。 |
| `frontend/pages/01_User_Profile_Query.py` | 用户画像查询及解释页面，支持局部刷新。 |
| `frontend/pages/02_Profile_Change.py` | 规则列表与增删改界面。 |
| `frontend/pages/03_System_Operations.py` | 系统运维面板：后台任务控制与训练触发。 |
| `frontend/utils.py` | 前端公共工具：后端地址解析、HTTP 请求封装。 |

## 数据流概览

1. **模拟数据生成**：`MockRealtimeAPI` 使用 Faker 生成初始用户/产品/应用/事件，并存入 SQLite。
2. **数据摄取**：`DataIngestionService` 预热用户缓存，监听实时事件，更新行为统计并写入事件队列。
3. **增量学习**：`IncrementalLearner` 消费事件批次，更新缓存嵌入，供混合画像服务调用。
4. **画像评分**：`HybridProfilingService` 执行规则、获取嵌入，输出最终得分、gate、规则详情等。
5. **解释产出**：`ExplainerService` 基于 SHAP 生成特征贡献向量与正负向 Top-K。
6. **全量刷新**：`RefreshOrchestrator` 调用 `GraphBuilder` 构图、`HGTTrainer` 训练、`HybridProfilingService` 融合训练，并刷新嵌入与解释缓存。
7. **API 暴露**：`profiling.py` 将各项能力以 REST 接口对外提供。
8. **前端展示**：Streamlit 页面消费 API，完成画像查询、监控与运维控制。

## 后端接口速览

- `GET /health`：返回设备模式、摄取状态与增量循环概览。
- `GET /api/v1/user/<user_id>`：获取缓存画像、行为统计、最后事件时间。
- `GET /api/v1/recommendation/<user_id>`：输出简单策略推荐。
- `GET /api/v1/explain/<user_id>`：返回 SHAP 解释与规则贡献。
- `POST /api/v1/graph/refresh`：触发全量刷新流程，可配置采样比例、训练轮次、学习率等。
- `POST /api/v1/operations/ingestion/*`：启动/停止数据摄取。
- `POST /api/v1/operations/incremental/*`：控制增量学习循环。
- `POST /api/v1/operations/fusion/train`：手动训练融合核心。
- `POST /api/v1/operations/training/hgt`：执行 HGT 遮蔽训练与评估。
- `POST /api/v1/operations/rules/refresh`、`/explainer/clear`、`/shutdown`：刷新规则结构、清空解释缓存、优雅停机。
- `GET /api/v1/operations/status`、`/metrics`、`/stream`：获取系统概览、当前指标与实时事件流。

## 前端页面说明

1. **总览页（`app.py`）**：展示健康状态、实时指标曲线、全量刷新表单及使用指南。
2. **画像查询页**：输入用户 ID 获取画像详情、SHAP 可视化、推荐列表，并支持单独触发刷新。
3. **规则管理页**：加载规则列表，提供新增、编辑、删除操作，统一错误提示。
4. **系统运维页**：控制摄取、增量循环、融合训练、HGT 训练及缓存操作，查看历史记录。

## 环境部署

1. 准备 Python 3.10 环境（建议 GPU + CUDA 12.1）。
2. 安装依赖（见 `backend/requirements.txt`、`frontend/requirements.txt` 或整合版本）。
3. 启动后端：
   ```bash
   export BACKEND_BASE_URL=http://127.0.0.1:5000
   python backend/app/main.py
   ```
4. 启动前端：
   ```bash
   streamlit run frontend/app.py
   ```

## 测试与校验

- 使用 `python -m compileall backend/app frontend` 快速检查语法。
- 推荐补充单元/集成测试，覆盖规则、训练与 API 行为。

## 运维提示

- 后端缓存大量模型与数据结构，更改配置后建议重启进程。
- 规则变更会自动触发嵌入刷新与 SHAP 缓存清理，但仍需注意耗时。
- 默认使用 GPU 训练，可通过 `UPS_DEVICE=cpu` 强制退回 CPU。*** End Patch

flowchart LR
  %% Top-level layers
  subgraph FRONTEND["Frontend (Streamlit UI)"]
    direction TB
    UI_App[/"app.py (Streamlit)"/]
    Page_Query["01_User_Profile_Query.py"]
    Page_Rules["02_Profile_Change.py"]
    Page_Ops["03_System_Operations.py"]
    Page_Data["04_Data_Configuration.py"]
  end

  subgraph API["API Layer (Flask Blueprint)"]
    direction TB
    BP[("Blueprint: /api/v1 (profiling.py)")]
    Health["GET /operations/status<br/>GET /operations/metrics<br/>/operations/stream"]
    OpsIngest["POST /operations/ingestion/start<br/>/stop"]
    OpsIncrement["POST /operations/incremental/start<br/>/stop"]
    OpsTrain["POST /operations/training/hgt"]
    OpsFusion["POST /operations/fusion/train"]
    RulesAPI["GET/POST /rules<br/>PUT/DELETE /rules/<rule_name>"]
    UserAPI["GET /user/<user_id>"]
    ExplainAPI["GET /explain/<user_id>"]
    RecAPI["GET /recommendation/<user_id>"]
    GraphAPI["GET /graph/statistics<br/>POST /graph/refresh"]
    DataConfigAPI["GET/POST /data/config<br/>POST /data/regenerate"]
  end

  subgraph BACKEND["Backend Services"]
    direction TB
    Controller[/"SystemController"/]
    Ingest[/"DataIngestionService"/]
    Incremental[/"IncrementalLearner"/]
    GraphBuilder[/"GraphBuilder (PyG)"/]
    Trainer[/"ModelTrainer / HGTTrainer"/]
    Fusion[/"HybridProfilingService (RuleEngine + FusionCore)"/]
    Explainer[/"ExplainerService (SHAP-like)"/]
    Orchestrator[/"RefreshOrchestrator"/]
    FeatureStore[/"Feature Store / Embedding Cache"/]
    RuleStore[/"RuleStore / RuleEngine"/]
    Metrics["Metrics & Runtime Status"]
    DB["SQLite (历史数据)"]
    MockAPI[/"MockRealtimeAPI (Faker) - 实时事件源"/]
    ModelFiles["Model artifacts / embeddings on disk"]
  end

  %% Data sources and storage
  MockAPI -->|historical & realtime events| DB
  MockAPI -->|realtime events| Ingest
  DB -->|historical load| Ingest
  Ingest -->|enqueue events| Incremental
  Incremental -->|update embeddings| FeatureStore
  Trainer -->|save embeddings| ModelFiles
  Trainer -->|trained model| ModelFiles
  Orchestrator --> GraphBuilder
  Orchestrator --> Trainer
  Orchestrator --> Fusion
  Orchestrator --> Incremental
  GraphBuilder --> Trainer
  GraphBuilder -->|hetero graph| ModelFiles
  Fusion -->|fusion params / gate| FeatureStore
  Fusion -->|uses rules| RuleStore
  Fusion --> Explainer
  Explainer -->|caches explanations| FeatureStore

  %% API <-> Backend wiring
  BP --> Controller
  BP --> Ingest
  BP --> Incremental
  BP --> GraphBuilder
  BP --> Trainer
  BP --> Fusion
  BP --> Explainer
  BP --> RuleStore
  BP --> Orchestrator
  BP --> Metrics

  %% API endpoints link to logical services
  UI_App -->|calls| BP
  Page_Query -->|calls| UserAPI
  Page_Query -->|calls| ExplainAPI
  Page_Query -->|calls| RecAPI
  Page_Rules -->|calls| RulesAPI
  Page_Ops -->|calls| Health & OpsIngest & OpsIncrement & OpsTrain & OpsFusion & GraphAPI
  Page_Data -->|calls| DataConfigAPI

  UserAPI -->|score_user| Fusion
  ExplainAPI -->|explain| Explainer
  RecAPI -->|get_recommendations| Fusion
  RulesAPI -->|manage rules| RuleStore
  OpsIngest -->|start/stop| Controller
  OpsIncrement -->|start/stop| Controller
  OpsTrain -->|trigger| Controller
  OpsFusion -->|trigger fusion train| Controller
  GraphAPI -->|trigger refresh| Controller

  %% Cross-cutting arrows and notes
  Controller --> Ingest
  Controller --> Incremental
  Controller --> Orchestrator
  Controller --> Metrics
  Metrics -->|exposed via| BP

  classDef service fill:#f9f,stroke:#333,stroke-width:1px;
  classDef storage fill:#efe,stroke:#333,stroke-width:1px;
  class Controller,Ingest,Incremental,GraphBuilder,Trainer,Fusion,Explainer,Orchestrator service;
  class DB,MockAPI,ModelFiles,FeatureStore,RuleStore storage;

