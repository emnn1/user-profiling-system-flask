# 用户画像与策略推荐系统 (Prototype)

一个集成规则引擎与图神经网络（HGT）的混合用户画像原型，包含 Flask 后端与 Streamlit 前端。

## 目录结构

- `backend/` 后端（Flask）：数据摄取、图构建、增量学习、规则融合、解释性输出等
- `frontend/` 前端（Streamlit）：健康检查、全量刷新、画像查询与运维控制
- `ARCHITECTURE.md` 架构与模块说明

## 环境要求

- Python 3.10+（建议 3.10/3.11）
- 可选：CUDA GPU（若设置 `UPS_DEVICE=gpu` 则需要可用的 CUDA 环境）

## 快速开始

### 1) 后端启动（Flask）

在项目根目录执行：

```powershell
# 安装依赖
pip install -r backend/requirements.txt

# 选择计算设备：gpu 或 cpu（默认 gpu，会在无 CUDA 时报错）
$env:UPS_DEVICE = "cpu"  # 或 "gpu"

# 启动后端（开发模式）
python backend/app/main.py
# 默认监听 http://localhost:5000 ，健康检查在 /health
```

### 2) 前端启动（Streamlit）

```powershell
# 安装依赖
pip install -r frontend/requirements.txt

# 指定后端地址（可选，默认为 http://localhost:5000）
$env:BACKEND_BASE_URL = "http://localhost:5000"

# 运行前端
streamlit run frontend/app.py
```

打开浏览器访问 Streamlit 提示的 URL（通常是 http://localhost:8501）。

## 常见问题

- 若使用 GPU：请确保 `torch` 对应的 CUDA 版本正确并且 `torch.cuda.is_available()` 返回为真。
- 若只想在 CPU 上跑：将环境变量 `UPS_DEVICE` 设置为 `cpu`。
- 前端无法连接后端：确认后端已启动，且 `BACKEND_BASE_URL` 指向可达地址，或在 `.streamlit/secrets.toml` 里设置 `backend_base_url`。

## 许可

当前仓库用于原型与学习目的，未指定开源许可。若需要开源/闭源许可，请根据需要添加 `LICENSE`。
