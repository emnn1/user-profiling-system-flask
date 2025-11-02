# 系统架构说明

## 总体视图

```
MockRealtimeAPI (SQLite) --> DataIngestionService --> IncrementalLearner --> HybridProfilingService --> Flask REST API --> Streamlit UI
                                   |                        |
                                   v                        v
                              RefreshOrchestrator --> HGTTrainer / FeatureEncoder / GraphBuilder
```

系统可分为三层：

1. **数据层**：`MockRealtimeAPI` 提供的模拟数据集与实时事件流（SQLite 持久化）。
2. **服务层**：构图、嵌入、训练、解释等核心 Python 服务，通过统一接口协作。
3. **展现层**：基于 REST API 的 Streamlit 多页面控制台。

## 后端组件

### 应用启动 (`backend/app/main.py`)
- 根据 `UPS_DEVICE` 与 CUDA 是否可用选择 GPU/CPU。
- 构造共享实例：`MockRealtimeAPI`、`DataIngestionService`、`RuleStore`、`RuleEngine`、`GraphBuilder`、`HeteroFeatureEncoder`、`HGTModel`、`HGTTrainer`、`IncrementalLearner`、`HybridProfilingService`、`ExplainerService`、`RefreshOrchestrator`、`SystemController`。
- 首次启动执行 HGT 训练与嵌入刷新，确保模型热身。
- 启动 `AsyncLoopRunner` 托管后台协程并注册 `profiling` 蓝图。
- 提供 `/health` 健康检查与进程结束清理。

### API 层 (`backend/app/api/profiling.py`)
- 路由前缀 `/api/v1`。
- **运维接口**：摄取、增量循环、融合训练、HGT 训练、全量刷新、指标查询、解释缓存清理、后端停机。
- **画像接口**：画像查询、推荐输出、SHAP 解释。
- **规则管理**：列出、新增、更新、删除规则，自动刷新融合结构并清空解释缓存。
- **图刷新**：`POST /graph/refresh` 校验参数后调用编排器，并同步特征编码缓存。
- 统一使用 JSON 与合适的状态码区分验证错误与资源缺失。

### 基础设施

| 模块 | 职责 |
| --- | --- |
| `core/async_runner.py` | 在线程中维持 asyncio 循环，支持 Flask 内调用异步协程。 |
| `data_source/mock_data_provider.py` | 生成/读取 SQLite 数据，提供异步事件迭代器并可写入实时事件。 |
| `graph_services/graph_builder.py` | 构建/采样 `HeteroData`，维护节点映射，支持增量事件写入。 |
| `ml_models/encoders.py` 与 `feature_store.py` | 为不同节点类型构建数值+类别特征编码。 |
| `ml_models/hgt_model.py` | HGT 模型主体与对比学习损失工具。 |

### 服务层

| 服务 | 关键方法 | 说明 |
| --- | --- | --- |
| `DataIngestionService` | `start/stop`、`_stream_events`、`get_user_profile`、`get_recommendations` | 管理异步事件队列，缓存用户画像与行为统计。 |
| `IncrementalLearner` | `register_events`、`refresh_all_embeddings`、`get_latest_embedding` | 消费事件更新嵌入，具体实现见文件。 |
| `HybridProfilingService` | `profile_user`、`train_fusion_core`、`refresh_rule_structure` | 运行规则引擎、融合嵌入，支持融合核心训练与规则结构变更。 |
| `ExplainerService` | `explain`、`clear_cache` | 构建 SHAP 基线并返回贡献明细。 |
| `ModelTrainer` | `train_on_graph`、`run_automated_training`、`evaluate_edge_splits` | 负责 HGT 训练、遮蔽评估与自动化流程。 |
| `RefreshOrchestrator` | `refresh_graph`、`run_training_workflow` | 统一调度图刷新、训练、嵌入更新与指标记录。 |
| `SystemController` | 摄取/循环控制、`trigger_refresh`、`train_fusion_core`、`run_hgt_training`、指标订阅 | 运维总控，维护历史记录与实时指标。 |

## 数据流详解

1. **数据播种与存储**
   - `MockRealtimeAPI` 检查 `mock_data.db` 是否存在数据，不足时使用 Faker 生成。
   - 历史事件写入 `events` 表，实时事件按需追加。

2. **画像缓存**
   - `DataIngestionService.start()` 分页加载用户，初始化计数器并启动 `_stream_events()` 协程。
   - 事件到达后更新用户 `event_counts`、`last_event_at`，同时入队等待增量学习。

3. **增量循环**
   - `SystemController.start_incremental_loop()` 创建 `_incremental_loop()` 协程，周期性取队列事件批并交由 `IncrementalLearner` 处理。
   - 循环过程持续更新运行状态与历史记录，并通过订阅队列推送指标。

4. **图构建与维护**
   - `GraphBuilder.build_graph_from_snapshot()` 可按采样比例构建 `HeteroData`，同时过滤事件端点。
   - `update_graph_from_events()` 将新事件转换为边并刷新存储，必要时重建 `HeteroData`。

5. **模型训练**
   - `HGTTrainer` 先调用 `HeteroFeatureEncoder` 生成特征张量，再执行对比损失训练。
   - 自动化流程会遮蔽部分边、生成负采样，并计算验证/测试指标。
   - 支持温度、学习率覆盖等自定义参数。

6. **混合评分**
   - `RuleEngine` 将条件表达式编译为谓词函数，输出 one-hot 特征与加权得分。
   - `FusionCore` 计算 gate 与神经网络得分，合成最终结果。
   - `HybridProfilingService.profile_user()` 返回最终得分、gate、规则详情、嵌入与原始特征。

7. **可解释性**
   - `ExplainerService` 从缓存用户构建 SHAP 基线，惰性初始化 `KernelExplainer`。
   - `explain()` 输出最终得分、规则分/模型分、Top-K 贡献等信息。

8. **统一编排**
   - `RefreshOrchestrator.refresh_graph()` 串联图构建、HGT 训练、融合训练、嵌入刷新与缓存清理，并记录指标。
   - `SystemController` 将事件写入 `_task_history`，同时更新 `_metric_snapshot` 供前端拉取。

## 前端架构

### Streamlit 总览页 (`frontend/app.py`)
- 设置页面布局、缓存策略与 Session 状态。
- 调用 `/health`、`/api/v1/operations/metrics`，绘制 CPU/内存/GPU 曲线与刷新状态。
- 提供全量刷新表单，可配置模式、采样比例、训练轮次、学习率等参数。
- 通过 `st.autorefresh` 实现 5 秒自动刷新。

### 画像查询页 (`pages/01_User_Profile_Query.py`)
- 从 secrets 或环境变量解析后端地址。
- 支持手动触发全量刷新，参数包括采样比例、温度、学习率覆盖、随机种子、是否保留时间戳等。
- 展示用户基础信息、行为统计、SHAP 贡献条形图与推荐列表。

### 规则管理页 (`pages/02_Profile_Change.py`)
- 对 `/api/v1/rules` 进行 CRUD 操作，统一错误提示。
- 通过 DataFrame 表格展示现有规则。

### 系统运维页 (`pages/03_System_Operations.py`)
- 获取 `/health` 与 `/api/v1/operations/status`，展示摄取/增量循环状态与历史记录。
- 以按钮/表单的形式控制摄取、增量循环、融合训练、HGT 训练、规则刷新、解释缓存清理与停机。
- 长耗时调用配合加载动画与结果输出。

### 公共工具 (`frontend/utils.py`)
- 封装后端地址解析、GET/POST 请求与错误处理，供多页面复用。

## 观测与指标

- `SystemController` 维护 `_metric_snapshot`，前端通过 `/api/v1/operations/metrics` 读取。
- `subscribe_metrics()` 将指标推送至队列，为 SSE 流提供实时事件。
- `metrics_utils.capture_resource_snapshot()` 捕获 CPU、RSS 内存、GPU 使用情况，便于前端可视化。

## 部署注意事项

- **硬件**：默认启用 GPU，若无 CUDA 可设置 `UPS_DEVICE=cpu`。
- **有状态组件**：大量单例缓存，不建议多进程 WSGI；若需扩展，请确保初始化与资源隔离。
- **存储**：SQLite 数据位于 `backend/app/data_source/generated_data`，需保证容器或宿主可写。
- **后台任务**：关闭服务前通过 `/operations/shutdown` 清理，确保事件循环与协程优雅结束。

## 后续扩展思路

- 替换 `MockRealtimeAPI` 为真实数据源或消息队列。
- 为 API 与前端补充鉴权机制。
- 扩展自动化测试覆盖率，纳入 CI。
- 容器化并集成 CI/CD，支撑生产部署。*** End Patch
