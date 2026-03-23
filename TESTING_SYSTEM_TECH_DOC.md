# 测试系统技术文档（用于迁移到大乐透项目）

最后更新：2026-03-13

## 1. 文档目标

本文档基于当前仓库已实现的“测试系统”全链路代码整理，覆盖：
- 核心业务逻辑（如何评估预测方法中奖效果）
- 代码结构与调用链（CLI / API / 前端）
- 参数与数据契约（请求、事件、报告格式）
- 实际运行方式（命令、接口、产物）
- 迁移到“大乐透项目”的可执行方案与实施清单

本文面向后续使用 Codex 进行二次实现，重点强调可复用架构和迁移风险。

---

## 2. 调研范围（已阅读文件）

### 2.1 测试系统核心
- `backend/tests/test_prediction_system.py`
- `backend/tests/test_examples.py`

### 2.2 预测执行与方法注册
- `backend/core/__main__.py`
- `backend/core/method_registry.py`
- `backend/core/analyzer.py`（与测试系统耦合的输出/中奖逻辑）
- `backend/services/data_processor.py`（分析数据切片）

### 2.3 API 与前端接入
- `backend/api/wechat/routes/testing.py`
- `backend/api/wechat/routes/methods.py`
- `backend/api/wechat/config.py`
- `backend/main.py`
- `frontend/src/api/testing.ts`
- `frontend/src/types/testing.ts`
- `frontend/src/views/Testing.vue`
- `frontend/src/router/index.ts`
- `frontend/src/utils/constants.ts`
- `frontend/nginx.conf`

### 2.4 相关但独立的评估功能（对比分析）
- `backend/services/prediction_comparison.py`
- `backend/api/wechat/routes/comparison.py`

> 说明：`prediction_comparison` 是“指定历史期号的多次对比评估”，不是本文测试系统主链路，但在大乐透迁移时可作为“严格回测”实现参考。

---

## 3. 系统总览

当前“测试系统”本质是一个**方法评估编排器**，不是直接训练模型：

1. 获取最新开奖号码作为评估目标；
2. 通过子进程调用预测 CLI（`python -m backend.core.analyzer predict ...`）；
3. 解析命令行文本输出中的号码；
4. 按双色球中奖规则判断奖级；
5. 采用“渐进/随机”策略重复测试；
6. 汇总方法表现并输出 JSON/TXT 报告；
7. 可通过 FastAPI 暴露同步接口与 SSE 流式接口；
8. 前端 `Testing.vue` 实时展示进度、日志、中奖记录、统计结果。

### 3.1 模块职责分层

- **评估编排层**：`PredictionTester`（测试策略、结果统计、报告输出）
- **预测执行层**：`backend.core.__main__.py` + `SSQAnalyzer`（真正生成号码）
- **方法元数据层**：`method_registry.py`（方法列表/分类/默认参数）
- **服务接入层**：`routes/testing.py`（HTTP + SSE）
- **展示层**：`Testing.vue`

---

## 4. 核心业务逻辑

## 4.1 基准开奖获取

`PredictionTester.__init__` 初始化时会读取 `data/ssq_data_all.csv` 第一行作为“最新开奖”。
当前数据文件按期号倒序，第一行为最新期。

输出结构：
```json
{
  "issue": "2026026",
  "date": "2026-03-10(二)",
  "red_balls": [2, 9, 16, 22, 25, 29],
  "blue_ball": 3
}
```

## 4.2 奖级判定规则（双色球）

`calculate_prize_level` 规则：
- 6红+1蓝 => 一等奖
- 6红+0蓝 => 二等奖
- 5红+1蓝 => 三等奖
- 5红+0蓝 或 4红+1蓝 => 四等奖
- 4红+0蓝 或 3红+1蓝 => 五等奖
- 仅蓝球中 => 六等奖
- 否则未中奖

## 4.3 预测调用机制

`call_prediction_method` 通过子进程执行：
```bash
python -m backend.core.analyzer predict --method <method> [--periods N] [--count N] [--explain]
```

失败时自动二次重试，追加降级参数：
```bash
--no-gpu --no-parallel
```

这使“方法可用性”更高，但也引入了执行时间不确定性（当前无超时）。

## 4.4 输出解析机制

`parse_prediction_output` 从 stdout 逐行查找同时包含 `红球:` 和 `蓝球:` 的行，然后解析号码。

兼容格式示例：
- `红球: 02 03 13 16 19 32 | 蓝球: 01`
- `红球: [03, 09, 15, 22, 27, 33] 蓝球: 12`

解析成功后统一成：
```json
{
  "method": "markov",
  "periods": 100,
  "count": 1,
  "predictions": [
    {"red_balls": [2,3,13,16,19,32], "blue_ball": 1}
  ],
  "raw_output": "..."
}
```

## 4.5 单次测试与多策略测试

### 单次测试 `test_single_prediction`
- 执行预测
- 对每注计算奖级
- 聚合 `best_prize` / `total_prizes`
- 写入内存结果 `test_results`
- 若有中奖写入 `winning_records`
- 若设置 `event_callback`，推送 SSE 事件（`result`/`winning`）

### 渐进策略 `test_method_progressive`
- 期数从 `start_periods` 到 `end_periods`，按 `step` 增长（默认 50）
- 每个期数做一次预测
- 达到 `target_prize` 立刻返回成功

### 随机策略 `test_method_random`
- 随机抽样 `periods` 与 `count`
- 测试 `test_count` 次
- 达到 `target_prize` 立刻返回成功

### 多方法测试
- 串行：`test_custom_methods`
- 并行：`parallel_test_methods`（ThreadPoolExecutor）

## 4.6 报告输出

`generate_report` 生成：
- `test_results/reports/test_report_<session>.json`
- `test_results/reports/test_report_<session>.txt`
- `test_results/logs/test_<session>.log`

JSON核心字段：
- `session_id`
- `test_time`
- `total_tests`
- `winning_tests`
- `winning_rate`
- `method_stats`
- `prize_stats`
- `best_methods`（当前代码未填充）

---

## 5. 命令行接口（测试系统）

入口：`python3 backend/tests/test_prediction_system.py`

## 5.1 参数

- `--method`：单方法 / 逗号分隔 / `all`
- `--methods`：显式多方法（优先于 `--method`）
- `--strategy`：`progressive` / `random`
- `--target-prize`：一等奖~六等奖
- `--periods-range`：`start:end`
- `--count-range`：`start:end`
- `--max-tests`：每方法最大次数
- `--parallel`：多方法并行
- `--workers`：并行线程数
- `--data-file`：开奖数据 CSV 路径
- `--results-dir`：报告输出目录

## 5.2 常用命令

```bash
# 单方法随机测试
python3 backend/tests/test_prediction_system.py \
  --method markov \
  --strategy random \
  --max-tests 30 \
  --periods-range 20:300 \
  --count-range 1:3 \
  --target-prize 六等奖

# 多方法并行测试
python3 backend/tests/test_prediction_system.py \
  --methods "super,lstm,monte_carlo" \
  --strategy random \
  --max-tests 50 \
  --parallel --workers 4 \
  --target-prize 四等奖
```

---

## 6. API 与 SSE 契约

API 前缀：`/api/wechat`

## 6.1 同步接口

### POST `/testing/run`
请求体（核心字段）：
```json
{
  "methods": ["markov", "lstm"],
  "strategy": "random",
  "target_prize": "四等奖",
  "periods_start": 10,
  "periods_end": 2000,
  "count_start": 1,
  "count_end": 5,
  "max_tests": 50,
  "parallel": true,
  "workers": 4
}
```

响应 `data` 与 CLI 报告结构一致，附加：
- `tested_methods`
- `successful_methods`
- `report_files`

### GET `/testing/options`
返回：
- `available_methods`（来自 `PREDICTION_METHODS`）
- `target_prizes`

## 6.2 流式接口

### GET `/testing/stream`
Query 参数与 `run` 对齐，服务端开启线程执行测试并推送 SSE。

事件类型：
- `log`
- `progress`
- `result`
- `winning`
- `complete`
- `error` / `error_event`（当前实现不统一，见风险章节）

前端通过 `EventSource` 监听上述事件并实时更新 UI。

---

## 7. 前端实现说明

`Testing.vue` 行为：

1. 页面加载后调用 `/api/wechat/methods` 获取方法列表并分组；
2. 选择参数后调用 `createTestingStream()` 建立 SSE；
3. 处理事件：
   - `progress`：更新当前方法和累计测试数
   - `winning`：追加中奖卡片
   - `result`：累计中奖注数
   - `complete`：展示汇总统计表
4. 可手动停止（关闭 EventSource）。

生产 Nginx 对 `/api/wechat/testing/` 单独配置了长超时（12 小时），用于支持长跑测试任务。

---

## 8. 方法注册与一致性机制

`backend/core/method_registry.py` 是唯一方法源：
- `PREDICTION_METHODS`：方法名 -> `SSQAnalyzer` 方法名
- `METHOD_DISPLAY_NAMES`：前端显示名
- `METHOD_CATEGORY_MAP`：分类
- `CATEGORY_DEFAULT_PARAMS`：默认参数
- `build_algorithm_config()`：构建 `ALGORITHM_CONFIG`

测试系统使用 `PREDICTION_METHODS.keys()` 作为可用方法列表，确保与 CLI 一致。

---

## 9. 实跑验证（2026-03-13）

已执行最小化命令：
```bash
python3 backend/tests/test_prediction_system.py \
  --method markov \
  --strategy random \
  --max-tests 1 \
  --periods-range 20:20 \
  --count-range 1:1 \
  --target-prize 六等奖
```

观察结果：
- 命令成功执行并生成报告；
- 产物目录按预期创建：`test_results/logs`、`test_results/reports`；
- JSON/TXT 报告字段与代码一致；
- 单次预测耗时约 20 秒（包含模型初始化、字体缓存等启动成本）。

---

## 10. 现状风险与已识别问题

以下为迁移前必须知晓的关键点：

1. **评估目标是“最新开奖”**
- 当前测试并非严格历史回测，而是把“最新一期”当目标。
- `get_analysis_data(periods)` 使用 `head(periods)`（最新到最旧），存在把目标期信息纳入分析窗口的可能，评估有数据泄漏风险。

2. **多方法场景下部分参数未生效**
- 当测试多个方法（串行或并行）时，`periods-range` / `count-range` 在当前实现中未透传到 `test_method_random` / `test_method_progressive`，会退回默认范围。

3. **输出解析依赖文本格式**
- 解析逻辑依赖“stdout包含 `红球:` 和 `蓝球:`”。
- 若 CLI 输出格式调整，测试系统可能全部解析失败。

4. **子进程无超时**
- `subprocess.run(... timeout=None)`，某些方法卡住时会无限等待。

5. **SSE 错误事件命名不一致**
- 后端有时发 `error`，有时发 `error_event`；前端仅显式监听 `error_event`，可能导致部分错误信息丢失。

6. **复现性不足**
- 随机测试未提供 seed 参数，难以复现某次结果。

7. **best_methods 字段未真正计算**
- 报告里存在 `best_methods`，但当前逻辑未填充值。

---

## 11. 迁移到大乐透项目的建议架构

建议不要直接复制当前实现，推荐抽象成“彩种无关评估框架 + 彩种规则插件”。

## 11.1 目标拆分

- **通用评估引擎**（复用）
  - 测试策略（progressive/random/parallel）
  - 预测执行（CLI/API/函数调用）
  - 事件流（log/progress/result/winning/complete）
  - 报告生成（JSON/TXT）

- **彩种规则层**（大乐透定制）
  - 号码范围与个数（前区/后区）
  - 奖级判定规则
  - 输出解析规则（建议 JSON 化）
  - 数据文件字段规范

## 11.2 建议新增抽象接口

```text
LotteryRule
- parse_prediction(payload) -> NormalizedTicket
- evaluate(ticket, winning) -> PrizeResult
- validate_ticket(ticket) -> bool
- format_ticket(ticket) -> str

PredictionRunner
- run(method, params) -> RawPrediction

TestEngine
- test_single(...)
- test_progressive(...)
- test_random(...)
- test_parallel(...)
- generate_report(...)
```

## 11.3 大乐透专属改造点

1. 号码结构：双色球是 `6红+1蓝`，大乐透通常是 `5前区+2后区`。  
2. 奖级规则：大乐透奖级判定与双色球不同，需独立实现。  
3. 数据源字段：如 `front_balls/back_balls`，需统一解析。  
4. 输出格式：建议预测 CLI 增加 `--json`，避免文本解析脆弱性。

---

## 12. 推荐实施路线（给 Codex）

### 阶段 1：抽出通用测试引擎
- 从 `PredictionTester` 拆出与双色球无关的逻辑；
- 注入 `LotteryRule`；
- 保留事件回调和报告输出结构。

### 阶段 2：实现大乐透规则插件
- 编写 `DltRule`：
  - 票据结构校验
  - 奖级判断
  - 显示格式

### 阶段 3：改造预测执行协议
- 大乐透预测命令统一返回 JSON（优先）；
- 测试引擎优先解析 JSON，文本作为降级。

### 阶段 4：迁移 API
- 复刻 `run + stream + options` 三个接口；
- SSE 事件名统一，仅保留 `error` 或 `error_event` 其中一种。

### 阶段 5：迁移前端
- 复用当前 Testing 页面交互模型；
- 替换字段名与奖级展示；
- 保留中奖卡片 + 方法统计 + 报告链接。

### 阶段 6：补齐测试
- 单元测试：奖级判定、输出解析、参数路由；
- 集成测试：单方法/多方法/并行/SSE；
- 回归测试：固定 seed 验证结果可复现。

---

## 13. 迁移验收清单

- [ ] CLI 单方法测试可运行并出报告
- [ ] 多方法并行测试稳定完成
- [ ] SSE 可实时展示 log/progress/result/winning/complete
- [ ] 奖级判定与大乐透规则一致
- [ ] 报告字段完整、可追溯（session_id、参数、方法统计）
- [ ] 无文本解析脆弱点（优先 JSON 协议）
- [ ] 关键流程有自动化测试覆盖

---

## 14. 结论

当前仓库“测试系统”已经具备可迁移的完整闭环（编排 -> 评估 -> 报告 -> API -> 前端），但它是以双色球规则和文本解析为前提的实现。迁移到大乐透时，建议将“评估引擎通用化 + 彩种规则插件化 + 预测输出 JSON 化”作为主线，以避免规则硬编码和解析脆弱性在新项目中重复出现。
