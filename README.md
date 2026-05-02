# Transformer-based VAV AHU Fault Detection

本项目为本科毕业设计提供一个可直接运行的完整代码框架：

1. 基于 AHU（空气处理单元）运行数据进行数据清洗与时序样本构造。
2. 使用 Transformer 编码器提取时变动态特征。
3. 在健康样本上建立统计量阈值（Hotelling T² + SPE）用于故障检测。
4. 进行故障类型分类（可选），并输出可视化图。

> 你需要准备并放置数据集到 `data/` 目录。项目已内置 `--generate-demo` 可生成可运行的模拟数据，方便先跑通流程。

## 推荐数据集（可选方向）

- ASHRAE RP-1312 / HVAC Fault Detection datasets（常用于 AHU/FDD 研究）
- 公开建筑能耗与 HVAC 运行数据（Kaggle/UCI/科研附录）

要求 CSV 至少包含：

- 时间列：`timestamp`
- 传感器列（示例）：`supply_air_temp`, `return_air_temp`, `outdoor_air_temp`, `supply_air_humidity`, `fan_speed`, `damper_position`, `cooling_valve`, `heating_valve`, `static_pressure`
- 标签列：`fault_label`（0=正常，1..K=故障类型）

你可以在 `config/default.yaml` 修改列名映射。

## 运行

```bash
python run.py --config config/default.yaml --generate-demo
```

使用真实数据：

```bash
python run.py --config config/default.yaml --data data/your_ahu.csv
```

输出目录：`outputs/<timestamp>/`

- `metrics.json`：检测与分类指标
- `thresholds.json`：T²/SPE 阈值
- `plots/*.png`：可视化曲线
- `model.pt`：训练好的模型

## PyCharm 使用建议

- 打开项目目录后，将 `run.py` 设为运行入口。
- Run Configuration 参数示例：
  - `--config config/default.yaml --generate-demo`
- 建议创建虚拟环境并安装：`numpy pandas torch scikit-learn matplotlib pyyaml`


## 已有 PyCharm 环境时怎么做（你的问题）

建议：**优先在原有环境内继续开发，不要先重建环境**。

### 推荐流程（最稳妥）
1. 先备份你现有项目目录（或用 Git 提交一次）。
2. 在原环境里新建一个分支/新目录，把本项目代码放进去。
3. 在原环境执行依赖检查：
   - `pip show numpy pandas torch scikit-learn matplotlib pyyaml`
4. 缺什么再补什么，不缺就不重装。
5. 在 PyCharm 里**新增 Python 文件**（或直接替换对应模块文件），不要动解释器配置。

### 什么时候才需要新建环境
- 你旧项目依赖冲突严重（例如不同 torch 大版本冲突）。
- 你想让毕设代码与旧项目完全隔离，便于复现实验。

### 实操建议
- 如果你现在的解释器已经是你列出的版本组合，通常直接复用即可。
- 你可以把本仓库当作“毕设主工程”，旧代码按模块复制进 `src/` 对比迁移。
- 先跑 `python -m py_compile run.py src/*.py` 做语法检查，再跑训练命令。

## 手把手详细步骤（从你当前 PyCharm 环境开始）

> 目标：不推翻你已有环境，在最小改动下把毕设项目跑通。

### 第 0 步：确认你现在用的是哪个解释器
1. 打开 PyCharm 项目。
2. 进入 `File -> Settings -> Project: xxx -> Python Interpreter`。
3. 记下解释器路径（例如 `.../venv/Scripts/python.exe`）。
4. 确认它就是你一直在用的那个环境，先不要切换。

### 第 1 步：备份当前代码（防止误操作）
1. 如果你用 Git：先 `commit` 一次当前代码。
2. 如果你不用 Git：复制一份项目目录到 `xxx_backup_日期`。
3. 目的：后面任何改动都可回退。

### 第 2 步：在原环境检查依赖，不要盲目重装
在 PyCharm 终端运行：

```bash
pip show numpy pandas torch scikit-learn matplotlib pyyaml
```

- 若某个包提示 `WARNING: Package(s) not found`，再安装它。
- 安装命令（缺哪个装哪个）：

```bash
pip install matplotlib pyyaml
```

> 你已装的包不要重复降级/升级，除非出现版本冲突。

### 第 3 步：创建项目目录结构
确保目录如下（没有就新建）：

```text
first-blood/
  config/
  data/
  src/
  outputs/
  run.py
```

### 第 4 步：准备配置文件
1. 打开 `config/default.yaml`。
2. 重点检查三项：
   - `csv_path`：你的数据 CSV 路径。
   - `feature_cols`：和你的真实列名一致。
   - `label_col`：故障标签列名（0=正常，1..K=故障）。
3. 如果列名不一致，必须在这里改成你数据的真实字段。

### 第 5 步：先用演示数据跑通（验证流程）
在终端运行：

```bash
python run.py --config config/default.yaml --generate-demo
```

预期结果：
- 控制台打印 `Done. Outputs:`。
- `outputs/时间戳/` 下有 `metrics.json`、`thresholds.json`、`model.pt`、`plots/`。

如果报错：
- `No module named matplotlib`：安装 `matplotlib`。
- `No module named yaml`：安装 `pyyaml`。

### 第 6 步：接入你的真实数据
1. 把 CSV 放进 `data/`（例如 `data/my_ahu.csv`）。
2. 运行：

```bash
python run.py --config config/default.yaml --data data/my_ahu.csv
```

3. 若报字段错误（如 `KeyError`），说明 `feature_cols`/`label_col` 与 CSV 列名不一致，回到第 4 步修正。

### 第 7 步：检查结果是否“可信”
1. 打开 `metrics.json` 看：
   - `classification.acc`
   - `classification.f1_macro`
   - `detection.binary_accuracy`
2. 打开 `plots/t2.png` 和 `plots/spe.png`：
   - 看故障时段是否明显越过阈值线。
3. 如果检测效果差：
   - 调 `sequence_length`（如 24、48、96 对比）；
   - 调 `d_model`、`num_layers`；
   - 检查标签质量（错标会严重影响结果）。

### 第 8 步：在 PyCharm 固化运行配置（避免每次手输）
1. 打开 `Run -> Edit Configurations`。
2. 新建 Python 配置：
   - Script path: `run.py`
   - Parameters: `--config config/default.yaml --data data/my_ahu.csv`
   - Working directory: 项目根目录
3. 保存后点击绿色三角直接运行。

### 第 9 步：什么时候“必须”新建环境
只有下面情况才建议新建：
1. 你当前环境和别的项目冲突严重（包版本互相覆盖）。
2. 你要做论文复现实验，要求环境完全可重建。
3. 你后续要上 GPU 且要单独配 CUDA 版 torch。

否则，**继续用原环境 + 新增 Python 文件**是最省时最稳的方案。

### 第 10 步：毕业设计交付建议（非常实用）
1. 每次改模型参数都记录到实验表（日期、参数、指标）。
2. 保留最优 `model.pt` 与对应配置文件快照。
3. 最终论文至少给出：
   - 数据说明（字段、采样间隔、故障类型）
   - 模型结构图
   - 阈值法原理（T²/SPE）
   - 对比实验（不同序列长度/不同模型）
