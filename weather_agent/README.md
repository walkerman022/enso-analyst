# weather_agent · 气象垂直 Agent（ENSO 监测 + 知识检索 + 多 Agent 会商）

基于本地 LightAgent v0.11.1 搭建。17 个业务工具（联网检索作为知识库的回退，默认关闭）。做两件事：**把 ENSO 指数查准**，以及**保证模型说出的每个数字都能对上工具出处与时间**。

## 快速开始

```bash
cd D:\AgentRoom\LightAgent-0.11.1\LightAgent-0.11.1
python -m weather_agent.cli "最近一期 ONI 是多少？现在是不是厄尔尼诺？"
python -m weather_agent.cli -i                    # 交互模式
python -m weather_agent.cli --session demo1 -i    # 跨进程保留对话
python -m weather_agent.council "现在厄尔尼诺到什么程度了？"   # 多 Agent 会商
python weather_agent/checks/check_stage1.py       # 148 项代码级验收
python weather_agent/checks/eval_set.py           # 16 条问答级评估（真实调用模型）
python weather_agent/checks/eval_council.py       # 2 条会商评估（每条约 4 次模型调用）
python weather_agent/checks/eval_knowledge.py     # 6 条知识问答评估（含跨语言）
python weather_agent/checks/eval_tools.py         # 20 条工具调用准确率
python weather_agent/checks/eval_stability.py     # 稳定性（5 题 × 3 次）
python weather_agent/checks/metrics_report.py     # 运行度量报表
python weather_agent/checks/eval_cache.py         # 缓存命中率与一致性
python weather_agent/checks/cache_report.py       # 缓存命中率
python weather_agent/checks/eval_prediction.py    # 预测 vs 观测的区分能力
python -m weather_agent.knowledge.build_index      # 构建/更新知识库索引
```

`--strict` 数值无出处时直接拦截；`--debug` 打印框架日志；`--session` 用 SQLite 持久化会话。

## 目录

| 文件 | 职责 |
|---|---|
| `agent.py` | 组装：身份、上下文预算、压缩阈值、预算上限、工具注册、输出护栏 |
| `prompts.py` | instructions（11 条：数字来源 / 季节口径 / 数据滞后 / 分型与图的规矩） |
| `tools/enso_fetch.py` | `fetch_oni` / `fetch_enso_index` / `describe_enso_state` / `find_oni_extremes` |
| `tools/nino_regions.py` | `fetch_nino_regions` / `classify_enso_type` / `list_enso_history` / `analog_evolution` / `plot_enso_trend` / `plot_enso_regions` |
| `tools/soi.py` | `fetch_soi`（大气侧，标准化 SOI） |
| `council.py` | 多 Agent 会商：海洋组 / 大气组 / 历史组 + 首席 |
| `evidence.py` | 证据账本：登记数值 + 数据源 + 所属时间（对象标签） |
| `number_check.py` | 数值回查 + 值-对象绑定 + 输出护栏 |
| `cache.py` | TTL 文件缓存（默认 6 小时） |
| `credentials.py` | 密钥读取（向上找 `LLM/.env`） |
| `metrics.py` | 运行度量：每次问答落一条记录（耗时/轮次/token/护栏/缓存命中） |
| `prompt_cache.py` | 提示缓存：键设计、TTL、标签失效、命中率统计 |
| `cached_agent.py` | 包装 Agent，命中直接返回、未命中调模型 |
| `tool_recorder.py` | 工具调用记录器（评估与副作用检测共用） |
| `prediction/` | 预测：模型注册表（models.json）+ 跨环境推理调用 |
| `tools/prediction.py` | `predict_enso` / `list_prediction_models` |
| `web_search.py` | 联网搜索（Bing HTML，默认关闭） |
| `tools/web_tools.py` | `web_search`（引用写 `[web:w1]`） |
| `memory.py` | 长期记忆：SQLite 偏好表，启动注入系统提示 |
| `tools/memory_tools.py` | `save_preference` / `recall_preferences` / `forget_preference` |
| `knowledge/` | 知识库：卡片 + 官方页面抓取 + SQLite 索引 + Qdrant 向量库 |
| `knowledge/embeddings.py` | 文本向量化（百炼，带本地缓存） |
| `knowledge/vector_store.py` | Qdrant 本地模式（embedded，不用起服务） |
| `tools/knowledge.py` | `search_knowledge`（返回片段 id，支持 `[kb:id]` 引用校验） |
| `summarizer.py` | 压缩用摘要器：禁止数字，失败退回确定性摘要 |
| `errors.py` | 数据源错误文案（`[LA-TOOL]` 前缀） |
| `output/` | 生成的 PNG |
| `data/` | 会话数据库（`--session` 时生成） |
| `checks/ci_check.py` | CI 门禁：跑断言并与基线对比 |
| `checks/check_stage1.py` | 代码级验收（148 项，普通脚本，不用 pytest） |
| `checks/eval_council.py` | 会商评估（结构 / 数值可追溯 / 关键词） |
| `checks/eval_knowledge.py` | 知识问答评估（引用存在 / 不编造引用 / 关键词 / 拒答） |
| `checks/eval_tools.py` | 工具调用准确率（20 条标注，BFCL 思路） |
| `checks/eval_stability.py` | 稳定性（5 题 × 3 次，核心数值一致） |
| `checks/metrics_report.py` | 度量报表（P50/P95 延迟、轮次、token、护栏率） |
| `checks/eval_cache.py` | 缓存实测（命中率、逐字一致、失效） |
| `checks/cache_report.py` | 缓存命中率报表 |
| `checks/eval_set.py` | 问答级评估（16 条，真实调用模型） |

## 预测能力

模型不是本项目训练的，而是接入论文实验权重（PhyDNet 系列，lead 条件化）：

```
models.json（模型注册表）→ inference.py（子进程调 dy_enso 环境）
  → PhyDNet-master/infer_once.py（加载权重、12 帧输入、自回归解码）
  → 场 → Niño3.4 区域平均 → JSON
```

三个关键设计：

- **跨环境子进程**：Agent 环境没有 torch，模型环境（dy_enso）有。进程隔离，模型崩了不影响 Agent；
- **预测与观测分离**：工具返回 `kind=forecast`，带 `issued_from`、`data_through`、`model_id`；
  提示词第 14 条禁止把预测说成"当前 ONI"；账本里预测值来源记为 `forecast/<model_id>`，与观测可区分；
- **指数口径**：模型输出是 Niño3.4 z-score，不是摄氏度，提示词第 15 条禁止换算或改称 ONI。

已知限制：当前 nc 数据截止 2026-02，比 ONI 观测（2026-10）落后约 9 个月；
模型对 2011/12 冷位相个例预测偏保守（RMSE 0.81 z，未抓住 La Niña 相位）。

## 输入护栏

拦三类硬边界（零误报）：空问题、超长（默认 4000 字符）、控制字符。
语义级注入检测刻意不做：误报不可控会把正常问题拦掉；数值类注入由输出护栏兜住，
知识类注入的实际风险面待注入测试集量化（审查报告 S3）。

## 联网回退（默认关闭）

`search_knowledge` 固定策略：**先查本地库，无命中且开关打开时才联网**。
开启：`--web` 或 `WEB_SEARCH_ENABLED=1`。

- 不做成独立工具，因为模型无法可靠判断“本地到底有没有”；策略内置，行为确定
- 只请求固定搜索域，不抓取结果页（不跟随用户可控 URL，规避 SSRF）
- 返回标题、链接、摘要（截断 180 字）与编号，引用写 `[web:w1]`
- `[web:w1]` 与 `[kb:def-oni]` 分开校验，编造编号会被护栏识别
- 网页内容标记为不可信输入；数字视为未核实，不能当观测值

## 提示缓存

两层，解决不同问题：

| 层 | 谁做 | 命中效果 | 我们能控制的 |
|---|---|---|---|
| 提供方前缀缓存 | 服务端 | 只减价，次数不变 | 保持 system 前缀稳定 |
| 应用层提示缓存 | 我们自己 | 直接不调模型 | 键设计、TTL、失效策略 |

应用层缓存的键包含五部分：模型名、系统提示哈希、完整消息哈希、**工具 schema 哈希**、
**环境版本（偏好更新时间 + 知识库版本）**。后两项是关键——工具改了、偏好改了、知识库重建了，
问题字面相同但答案已经不同，只按问题做键必然命中脏数据。

失效三条路：TTL（默认 10 分钟，兜住数据更新）、标签失效（偏好变化清 `prefs`、索引重建清 `kb`）、
版本号进键（即使忘了调失效，键也变了）。

不缓存的三种情况：带 `session_id` 的续聊（历史在框架内部，键算不稳）、调用了写操作工具
（`save_preference` 等有副作用）、回答以 `[LA-` 开头（故障结果不固化）。

实测：同一问题连问两次，第二次命中、逐字一致、token 增量为 0；换问题未命中；
改偏好后同问题重新未命中。

## 评估与度量

四层，从"代码对不对"到"跑起来什么样"：

| 层 | 脚本 | 内容 | 当前 |
|---|---|---|---|
| 代码级 | `check_stage1.py` | 解析、绑定、隔离、压缩、记忆、向量、度量 | 83 项全过 |
| 问答级 | `eval_set.py` / `eval_knowledge.py` / `eval_council.py` | 数据、知识、会商 | 16/16、6/6、2/2 |
| 能力级 | `eval_tools.py` | 工具调用准确率、平均轮次与 token | 20/20，2.05 次调用，1.4 次工具 |
| 稳定性 | `eval_stability.py` | 同题 3 次，核心数值是否一致 | 5/5 |

运行度量单独一层：`metrics.py` 在每次问答后写一条记录，`metrics_report.py` 输出
P50/P95 延迟、平均轮次、平均 token、护栏触发率、引用比例。这是"线上表现"的依据，
和前面四层（离线质量）互补。

## 多 Agent 会商

```
海洋组（ONI/分区/分型）─┐
大气组（SOI）──────────┼→ 首席（无取数工具，只读三份报告）→ 纪要
历史组（事件/类比演变）─┘
```

**为什么值得多 Agent**：三套判据需要三份独立上下文；单 Agent 看到 ONI 2.46 之后，后面全往"强厄尔尼诺"靠，不会主动说"海洋和大气不一致"。

**账本隔离**：每个角色一个命名空间（`ocean` / `atmosphere` / `analog`），工具代码不用改，编排层 `evidence.use_scope()` 切过去。首席的账本是三组并集（`merge_into("council", ...)`），所以它引用任何一组的数值都算有出处。

**首席不给取数工具**（`tools=[]`），从机制上禁止它自己另查一遍；它只剩框架内置的 `safe_expression` 和 `upload_file_to_oss`。

## 四条设计决定（都是踩坑后改的）

**1. 数值回查靠账本，不靠对话历史。**
LightAgent 的输出护栏只收到 `{"agent_name", "trace_id"}`（`core.py:1432`），拿不到 `chat_params`。所以工具在返回前把数值写进 `evidence`，护栏拿账本比对。账本按进程清空，不按轮次——否则多轮追问引用上一轮数值会被误判成编造。

**2. 常量与派生量必须放行，否则误杀一片。**
`Niño3.4`、`±0.5`、`5°N-5°S` 是术语不是数据；`2.16-0.36=1.80` 是合法推算。两者都放行，只拦"凭空出现的小数"。实测浮点误差：`2.45-2.4=0.05000000000000027`，容差必须加 `1e-9`。

**3. 工具返回体积要自己控，不能靠压缩兜底。**
传 920 个季节（约 10 万字符）会被 `ContextCompactor` 溢出成哈希，模型只收到 `[spilled] sha256...`。现在 `fetch_oni` 单次最多 100 季（约 6.4k 字符）；查历史峰值用 `find_oni_extremes`；`list_enso_history` 内部直接读全序列，不走这个 100 季上限。

**4. 值-对象绑定：光有出处不够，还要"挂对时间"。**
账本给每个数值打了时间标签（`JAS 2026`、`2026-06`）。回答里句内若只有一个时间标签，就检查数字是否属于它；若数字真实存在、但属于别的时间，判为"时间错配"并提示。实测能抓住"`NDJ 2015 峰值 2.59`"被写成"`2.46`"这类错误。**只对小数生效**，整数（序号、月份、季数）会大面积误报。

## 数据与口径

| 数据 | 源 | 频率 | 实测滞后 |
|---|---|---|---|
| ONI（Niño3.4 3 月滑动） | `cpc.ncep.noaa.gov/.../oni.ascii.txt` | 月 | 约 1 个月 |
| 四分区距平（Niño1+2/3/4/3.4） | `.../ersst5.nino.mth.91-20.ascii` | 月 | 约 4 个月（比 ONI 慢） |

两边滞后不同步：分型可能基于 6 月，而 ONI 已到 8-10 月窗口。`prompts.py` 第 9 条要求模型必须写明这个时间差。

## 已知限制

- **SST 距平场图没做**：需要网格数据（NetCDF），当前环境缺 `xarray`/`netCDF4`。用四分区柱状图替代。
- **分型阈值是经验值**：`|Niño3 - Niño4| < 0.2°C` 算过渡型，CPC 没有官方判据，代码里标了 `【推测】`。
- **错配校验有取舍**：只在“同一字段、不同时间”时判错配；若模型把 SOI 值写成 ONI 值（字段不同），不会报错——这是为了避免大面积误报。
- **摘要约束**：摘要里禁止出现阿拉伯数字，否则会改写数值、破坏证据回查；代价是摘要不能带具体数据，只能写意图与范围。
- **结构化输出没接**：`run()` 没有 `response_format` 参数，值-对象绑定靠句内标签启发式，跨句、多标签的句子会跳过检查。
- **账本是进程级全局变量**：多用户并发会串，做服务要按 session 分桶。
- **长期记忆只做偏好层**：SQLite 键值表 + 启动注入，不做语义检索；情景记忆和流程记忆没有表结构。
- **检索是混合式**：Qdrant 向量（1024 维，余弦阈值 0.42）+ 中文二元组词频，RRF 融合。
  向量解决跨语言（中文问、英文语料），词频保住精确术语（Nino3.4、SOI）。
- **依赖**：`qdrant-client>=1.9`（本地 embedded 模式，不需要 Docker）；向量化走百炼 `EMBEDDING_*`，未配置时自动退回纯词频。
- **官方页面抓取只取正文段落**，保留来源链接与抓取时间，未做增量更新调度。
- **RAG 已接入**（此前只做实时数据，现已补知识检索）：引用格式 `[kb:id]`，护栏校验该 id 是否真被召回。：数据是结构化+实时，走 API 更准。

## 下一步（候选）

台风（IBTrACS）、短临（数据门槛最高）、把分型改成多指标综合（SOI、信风）、结构化输出改造。
