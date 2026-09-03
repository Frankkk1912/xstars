# Plan: qPCR log-space 统计修正同步到 Excel/main（1c19a63 回灌）

- 状态：**已批准**（用户 2026-09-03 批准 rev 3 并指定唯一修订 p-value 标签；rev 4 仅落实该修订）
- Rev：4（2026-09-03；rev 2 实证修正，rev 3 新增 R11/R12，rev 4 标签定稿，见 Changelog）
- 目标基线：`main` @ `5f4c4099bd7c578e8d3fde9e59de0ef77a4dc3b9`
- 行为来源：`1c19a6334c84666206c034e50385d44911932a9f`（仅存在于 `feature/wps-support`）
- 依据：`plans/explore-20260903-qpcr-log-space-excel-sync.md`（下称 explore 报告）+ 用户访谈 2026-09-03

---

## 1. Goal

在 Excel/main 发布线上，使 qPCR 分析与 `1c19a63`（WPS 分支）行为完全等价：

1. qPCR 假设检验在 log2 fold-change（−ΔΔCt）线性空间执行——正态性/方差齐性判断、检验分支选择与 p 值均基于 log2 空间，而非 `2^-ΔΔCt` 的非线性 ratio 尺度；
2. qPCR 柱状图（BAR_SCATTER）使用几何均值作为柱高，SEM/SD/CI95 误差在 log2 空间计算后旋回 fold-change 轴，形成不对称误差条；
3. 写回 Excel 的处理后数据与绘图数据**仍是正值 fold change**，用户可见输出不变为 log2FC；
4. main 的全部 5 个 qPCR-capable 函数结构 / 6 个 `StatsEngine` 调用表达式一致采用该口径，ELISA 与 WB labeled 统计保持完全不变；
5. 以 pytest 全量通过作为验收，测试覆盖 log-space p 值（手算 Tukey 对拍）与几何均值/不对称误差条。

用户价值：消除同一 qPCR 数据在 Excel 入口与 WPS 入口得到不同统计/图形口径的双轨状态；统计检验空间符合 Prism/ΔΔCt 惯例（在 ΔCt 空间检验、报告 fold change）。

## 2. Requirements

| # | 需求 | 类型 | 来源 |
| --- | --- | --- | --- |
| R1 | 采用路径 A：从 `main`（5f4c409）新建 feat 分支手工移植；**不引入** WPS 的 `application/` 层、contracts、`export.py` | 硬约束 | 访谈 2026-09-03 决策 1 |
| R2 | 与 1c 完全等价的两部分行为：① 统计在 log2（−ΔΔCt）空间；② qPCR 柱状图几何均值 + log 空间 SEM/SD/CI95 不对称误差条。两部分都必须交付 | 硬约束 | 访谈 2026-09-03 决策 2 |
| R3 | 入口全覆盖：main 的 6 个 qPCR-capable `StatsEngine` 调用表达式（main.py :508-509、:667-668、:781-782、:842-843、:1477-1478、:1499-1500）全部经过 qPCR gate；ELISA（main.py :330-331）与 WB labeled（main.py :587-588）保持不变 | 硬约束 | 访谈 2026-09-03 决策 3；调用点清单=explore 报告 §1.4 |
| R4 | 严格复刻 1c 行为。~~已知 MINOR（xticklabels）不在本次修复；:804-826 测试不移植~~ **[superseded rev 2]**：rev 1 前提失效——实证表明 1c19a63 原版 `_qpcr_bars` 已含 `ax.set_xticks`/`ax.set_xticklabels`（1c plot_engine.py L173-174），tick-label 回归测试 :804-826 属于 1c 自身内容（review MINOR 已在 squash 提交中修复）。故"严格复刻 1c"= 原样移植含 tick-label 行为，并把 :804-826 测试意图移植进 `tests/test_plot_engine.py`（T4.2） | 硬约束 | 访谈 2026-09-03 决策 4 + rev 2 实证修正（见 §4.6-2） |
| R5 | 测试落点：扩展 `tests/test_presets.py`（log-space p 值 integration，含手算 Tukey/scipy 对拍，移植自 WPS test_presets.py :507-531 与 QPCRStatsSpaceTests :559-666 的可适用部分）+ `tests/test_plot_engine.py`（几何均值/不对称误差条，移植自 :649-666 意图）。不新建测试文件、不复制 WPS application 测试文件 | 硬约束 | 访谈 2026-09-03 决策 5 |
| R6 | 数值边界沿用 1c 当前行为：不新增 0/inf/非有限值 guard；残余风险记录在 Plan 风险区 | 硬约束 | 访谈 2026-09-03 决策 6 |
| R7 | 验收 = pytest 全量通过；真实 Excel 人工验证后置（记录为后续动作，非本 Plan 验收项） | 硬约束 | 访谈 2026-09-03 决策 7 |
| R8 | 不变量：写回/绘图数据仍是正值 fold change；仅假设检验输入与 qPCR bar 的中心/误差计算进入 log2 空间 | 硬约束 | 需求陈述 + explore 报告 §1.3 关键不变量 |
| R9 | stats gate helper 所在模块不依赖 xlwings，保证 pytest headless 可导入 | 硬约束 | 设计提示（pytest headless 可导入）+ explore 报告 §3.3（main.py 模块级 import xlwings 的代码库证据） |
| R10 | `StatsEngine` 保持通用，不感知 preset（统计空间选择发生在调用方） | 硬约束 | explore 报告 §3.1 既有分层约定 |
| R11 | qPCR 输出标签：① qPCR 路径的 processed data 标题追加后缀，变为 `Processed Data (2^-ΔΔCt)`（labeled 变体：`Processed Data — {gene} (2^-ΔΔCt)`）；② qPCR 路径 stats 表的 `p-value` 列重命名为 `p-value(−ΔΔCt)`（rev 4 定稿措辞，用户指定）。**仅限 qPCR 调用点**，ELISA/WB 的标题与列名保持不变；标签常量/辅助函数放 `presets/qpcr.py`（headless 可测）。WPS 侧（application/analysis.py）同名标签为后置对齐动作，不在本 PR | 硬约束 | 用户 2026-09-03 追加指令（rev 3） |
| R12 | 模板示例数据修正：`XSTARS_Templates.xlsx` qPCR sheet 的 TNF-a Control Ct 由 `[28, 28.2, 27.8]` 改为 `[28, 28.4, 27.7]`（仅 Q11/Q12 两格），使 control 组方差 ≠ 0，示例不再落入 Dunn's test（已预验证：走 ANOVA→Tukey，control fold-change 方差 0.0109）；仅修改被 git 跟踪的 `XSTARS_Templates.xlsx`，本地未跟踪的 "copy" 变体不动 | 硬约束 | 用户 2026-09-03 追加指令（rev 3）；预验证见 Changelog rev 3 |

## 3. Non-goals

本次明确**不做**：

1. **不引入 WPS `application/` 层、contracts、`xstars/application/export.py`**（路径 A 边界，R1）。
2. **不做 PR #1（`feature/wps-support` 整体，50 commits）的合并或预演**——本 Plan 与 PR #1 互相独立。
3. ~~**不修复 `_qpcr_bars` 的 xticklabels MINOR**，也不移植对应回归测试（R4）~~ **[superseded rev 2]**：实证表明 1c19a63 已含 tick-label 修复与回归测试，按"严格复刻 1c"随 T3.2/T4.2 正常移植，无独立 MINOR 修复项；如需另开 issue/PR 仅适用于未来新问题。
4. **不新增数值 guard**（0/inf/非有限值处理沿用 1c 行为，R6）。
5. **不做真实 Excel 人工验证**（普通/labeled qPCR、Quick、export include-stats、三种 error-bar 配置的真实宿主验证后置，见 §8 第 7 项）。
6. **不修改 `xstars/stats_engine.py`**（引擎不感知 preset，R10）。
7. **不新建测试文件、不复制 `tests/test_application_analysis.py`**（R5）。
8. **不对 `feature/wps-support` 分支做任何改动**（本工作全部落在新 feat 分支）。
9. 不新增配置开关/UI 项：qPCR 判定复用既有 `experiment_preset` 字段（explore 报告 §3.5）。
10. 不改 violin/line 等其他 ChartType 的绘图路径（1c 只特化 BAR_SCATTER）。

## 4. Research summary

### 4.1 代码库现状（锚点均复用 explore 报告已验证取证）

- **main 当前 qPCR 数据流**：`QPCRPreset.transform()` 输出 `2^-ΔΔCt` fold change（xstars/presets/qpcr.py:54-84）；labeled 路径 `transform_labeled()` 输出 per-target fold-change frame（qpcr.py:86-157）。main 把该 frame **原样**传给 `StatsEngine.analyze()`，无 qPCR 特判（explore 报告 §1.2）；`PlotEngine._bar_scatter()`（main plot_engine.py:79-118）走通用 seaborn bar plot，均值/误差条在线性 fold-change 轴计算；通用 `_error_value` 位于 main plot_engine.py:336-352。
- **1c/WPS 目标数据流**：`_stats_input_frame()`（WPS application/analysis.py:234-251）仅在 `isinstance(preset, QPCRPreset)` 时 `np.log2(transformed)`，否则原样返回；公开 config 包装 `stats_input_frame()`（:254-261）。`log2(2^-ΔΔCt) = -ΔΔCt` 精确成立，无需额外 preset 状态。绘图仍接收 fold-change frame；BAR_SCATTER + qPCR 时 `_bar_scatter()` 分发到 `_qpcr_bars()`（WPS plot_engine.py:85-127），后者经 `_qpcr_geo_stats`/`_log_error_value`（:418-456）在 log2 空间算中心/误差再以 2 的幂旋回线性轴（:129-174）。非 qPCR 走原 seaborn 路径。
- **main 调用点全景**（explore 报告 §1.4，已验证）：8 个 `StatsEngine` 创建/analyze 表达式、7 个函数；其中 6 表达式/5 函数 qPCR-capable（见 R3），ELISA `:330-331`（`_run_elisa_impl`，定义 :271）与 WB labeled `:587-588`（`_run_wb_labeled`，定义 :557）明确非目标。
- **六文件差异**（explore 报告 §1.5）：1c 修改 6 路径，其中 3 个（application/analysis.py、application/export.py、tests/test_application_analysis.py）在 main 不存在 → 直接 cherry-pick 必然 modify/delete 冲突，**不可行**。共同 3 文件（tests/test_presets.py 70 diff lines、main.py 705、plot_engine.py 377）结构分歧显著，需手工映射移植。`git merge-tree --write-tree main 1c19a63` 退出码 0 只证明"整体并入 WPS 历史"无文本冲突（main 是 WPS 严格祖先），不能解读为单 commit patch 干净。
- **main 测试布局**：tests/test_presets.py（711 行，qPCR→stats integration 在 :522 但仍直接分析 transformed fold change）、tests/test_plot_engine.py（:15-81 仅通用 bar/error/options，无 qPCR 图形测试）、tests/test_stats_engine.py（:33-156 通用引擎）。main 无 test_application_analysis.py。

### 4.2 外部调研

**未开展外部调研，理由**：本次为同仓库内部回灌——目标行为已在 `1c19a63` 完整实现并有测试覆盖（test_application_analysis.py:559-826、test_presets.py:507-531），统计口径依据（Prism/ΔΔCt 惯例：在 ΔCt 空间检验、报告 2^-ΔΔCt）已在 1c 的 `_stats_input_frame` docstring（application/analysis.py:234-251）中记录；无外部库选型或第三方最佳实践需求。

### 4.3 推荐方案：路径 A（从 main 新建 feat 分支手工移植）

**理由**（explore 报告 §4 + 访谈决策 1）：

- 范围可控：只交付 qPCR 行为修正，不耦合 WPS 大 PR 的 50 commits 与其未完成的合并决策（路径 B）；不延迟问题（路径 C 只在 WPS 分支叠提交，main/Excel 得不到修复）。
- cherry-pick 不是捷径：3 个 modify/delete 冲突 + 共同文件大幅分歧；cherry-pick 仅可作为提取 patch/审阅来源，行为移植以手工为准。
- Git 整合成本低不等于行为风险低：路径 B 的 fast-forward 会把 WPS 整体范围（application/contracts/worker/installer）一并带入，违背本次范围边界。

### 4.4 替代方案及取舍

- **路径 B（整体合并 feature/wps-support）**：长期最少重复（Excel/WPS 共用 application gate），但短期扩大范围显著、耦合 WPS 发布时机与验收；仅当 PR #1 已获批立即合并时适用——访谈已确认本次不采用。
- **路径 C（继续在 WPS 分支改再"同步"）**：表面改动最少，实质延迟问题；从该分支向 main 开 PR 会携带全部 WPS 历史，扩大 review 面；再摘单 commit 又回到路径 A 的结构冲突。不推荐。

### 4.5 stats gate helper 落点定稿：`xstars/presets/qpcr.py`（planner 决策）

**定稿**：在 `xstars/presets/qpcr.py` 模块级新增 helper（具体 API 见 T2.1）。

**理由**：

1. **满足 R9**：已验证该文件只 import `numpy`/`pandas`/`..config`/preset 基类（qpcr.py:1-14），无 xlwings；`xstars/main.py` 模块级 import xlwings，若 helper 落在那里，`tests/test_presets.py` 引用 helper 会拉入 xlwings，破坏 pytest headless 可导入性。
2. **qPCR 语义集中**：与 `QPCRPreset` 同处，gate 本质是 qPCR 特有的统计空间选择；且 `register_preset(ExperimentPreset.QPCR)`（qpcr.py:27）使 preset-instance 判定与 config 判定天然等价。
3. **无循环导入**：qpcr.py 已 `from ..config import ExperimentPreset`（qpcr.py:12），新增 helper 无需新依赖。
4. **未来去重友好**：命名对齐 WPS `stats_input_frame`（application/analysis.py:234-261），PR #1 合并时两处 gate 的收敛点语义邻近、易识别。

### 4.6 证据缺口（显式标注，不确定项）

1. **共同文件 hunk 级冲突未实测**（explore 报告 §1.5）：本 Plan 走手工移植，不依赖冲突解决，该缺口不阻塞；但实施时禁止用 cherry-pick 结果直接落盘（见风险 R2）。
2. **~~`_qpcr_bars` 的 1c 版本与当前工作区存在表面矛盾~~【rev 2 已实证解决】**：rev 1 曾推断工作区已含 1c 之后的 tick-label 补充修复。实证结论（2026-09-03，父 Agent `git show 1c19a63` + `git diff 1c19a63 -- <file>`）：① 1c19a63 原版 `_qpcr_bars` **已含** `ax.set_xticks(x_positions)`/`ax.set_xticklabels(groups)`（1c plot_engine.py L173-174）；② `git diff 1c19a63 -- xstars/plot_engine.py` 与 `git diff 1c19a63 -- tests/test_application_analysis.py` 均为空，且 `git log 1c19a63..feature/wps-support -- xstars/plot_engine.py` 为空——1c 后无任何 plot_engine 变更，工作区 = 1c 原版；③ tick-label 回归测试（:804-826，"Fix 3c"）属于 1c19a63 自身。**结论**：rev 1 的 MINOR 前提（访谈决策 4 描述的"缺 set_xticklabels"）描述的是 PR review 中间态，已在最终 squash 提交中修复。移植基准 = 1c19a63 原版，含 tick-label 行为与测试意图；无 post-1c 漂移风险。此为事实实证，非产品决策变更（"严格复刻 1c"决策不变，其含义随实证自然确定）。
3. **main 侧代码体不可从当前工作区直接读取**（工作区在 WPS 分支）：main.py / plot_engine.py 的行号锚点全部复用 explore 报告已验证锚点，未虚构任何 main 侧行号；实施时在 feat 分支上直接读取 main 版文件核对锚点后再改。

## 5. Gap analysis

| ID | 功能缺口 | 现状 | 影响 | 补齐任务 |
| --- | --- | --- | --- | --- |
| G1 | qPCR 假设检验缺少 log2（−ΔΔCt）空间 gate | main 无任何 log2 特判（explore 报告 §5：`git grep` main 无 `_stats_input_frame/log2` 命中），qPCR-capable 调用直接 `analyze` fold-change frame | 正态/方差决策、检验分支、p 值基于非线性 ratio 尺度，与 1c/WPS 口径不一致，统计结论可能错误 | T2.1、T2.2、T2.3 |
| G2 | 入口未全覆盖：6 个调用表达式需一致经过 gate | main 有 5 个 qPCR-capable 函数结构 / 6 个调用表达式（§1.4），分散于 preset/labeled/Run/Quick/export 两分支 | 只修单一入口会产生"同一数据不同入口不同结果"的双口径；export/include_stats 重算路径尤其易漏 | T2.2、T2.3、T5.1 |
| G3 | 非 qPCR（ELISA/WB）不变性无集中保障 | gate 若散落手写 `np.log2` 易误伤 ELISA（:330-331）/WB（:587-588） | 引入集中 helper + 明确非目标清单可消除误伤；缺测试时无回归保护 | T2.1（isinstance/config 谓词集中）、T4.1（passthrough 测试） |
| G4 | qPCR 柱状图缺少几何均值/log 空间不对称误差条 | main `_bar_scatter`（:79-118）只有通用 seaborn 算术均值/线性误差；无 `_qpcr_bars`/`_is_qpcr`/`_qpcr_geo_stats`/`_log_error_value` | 若只改统计不改图，图上中心趋势/误差与被检验空间视觉不一致；柱高与 p 值口径分裂 | T3.1、T3.2 |
| G5 | 测试保护缺失 | main test_presets.py 的 qPCR integration（:522）仍直接分析 transformed frame，未证明 log2 p 值；test_plot_engine.py（:15-81）无 qPCR 几何均值/不对称误差测试 | 移植后无回归保护，未来重构可能静默回退到线性空间口径 | T4.1、T4.2 |
| G6 | 移植基准证据不完整 | ~~`_qpcr_bars` 的 1c 原版内容与当前工作区存在差异（§4.6-2）~~【rev 2 已实证：1c19a63 原版含 set_xticklabels，工作区与 1c 零差异，基准无漂移】；cherry-pick 不可行仍需以手工移植规避（§4.1） | 基准不清可能导致误把 post-1c 修复一并移植或遗漏 1c 自身内容；rev 2 后残余影响仅限"误引 WPS application 范围" | T1.2（支撑性任务：锁定 1c 移植基准并记录实证结论，支撑 R4 与路径 A 边界）；T1.1（支撑 R1/R7：基线绿证明） |

**自查：无孤立缺口**——G1→T2.1/T2.2/T2.3；G2→T2.2/T2.3/T5.1；G3→T2.1/T4.1；G4→T3.1/T3.2；G5→T4.1/T4.2；G6→T1.1/T1.2。所有任务均有对应缺口：T1.x 支撑 G6（及 R1/R7 横切约束），T2.x→G1/G2/G3，T3.x→G4，T4.x→G5/G3，T5.x→G2 静态核查与 R7 验收。

## 6. Milestone 表格

| Milestone | Status | Dependencies | Validation | Notes |
| --- | --- | --- | --- | --- |
| M1 分支建立与移植基线 | [ ] | 无 | feat 分支自 5f4c409 创建；main 基线 pytest 全绿（或已知失败已记录区分）；1c19a63 参考文本提取完成且 `_qpcr_bars` 版本基准已实证记录（rev 2 已预实证，T1.2 仅需落档确认）；模板数据修正完成（T1.3） | §4.6-2 已实证（rev 2），T1.2 转为落档确认；T1.3 独立于分支基线也可先做 |
| M2 统计 gate 移植与 6 调用点接入 | [ ] | M1 | 静态核查：6 个调用表达式均经 `stats_input_frame`，ELISA/WB 两调用点无 gate；`np.log2` 仅存在于 helper；tests/test_stats_engine.py 未修改且全绿；qPCR 输出标签就位且 ELISA/WB 标签不变（T2.4） | helper 落点 presets/qpcr.py（§4.5）；R11 标签同文件落地 |
| M3 plot_engine qPCR 图形移植 | [ ] | M1（与 M2 可并行） | main plot_engine.py 含 `_is_qpcr`/`_qpcr_geo_stats`/`_log_error_value`/`_qpcr_bars` 且 BAR_SCATTER 分发生效；非 qPCR 路径原样；既有 TestBarScatter/TestErrorBars 全绿 | 严格复刻 1c 原版（rev 2 实证：含 set_xticklabels，基准=1c19a63，无 MINOR 残留） |
| M4 测试移植 | [ ] | M2、M3 | 新增测试全部通过；未新建测试文件；含 tick-label 回归测试（:804-826 意图，rev 2 修订） | 落点仅 test_presets.py + test_plot_engine.py（R5） |
| M5 全量验证与 Draft PR | [ ] | M4 | pytest 全量 0 failed；§8 静态核查清单全过；Draft PR 已开（base=main） | 验收标准=R7 |

Milestone 总数：5（≤10，符合目标 ≤7）。

## 7. 分 milestone 的 To-do checkbox 清单

### M1 分支建立与移植基线

- [ ] T1.1 从 main（5f4c409）新建 `feat/qpcr-log-space-excel-sync` 并确认基线绿
  - 文件：无（git 操作；不产生代码改动）
  - 修改：`git switch -c feat/qpcr-log-space-excel-sync 5f4c409`（自 main tip）；在分支上运行 `python -m pytest tests -v` 记录基线结果；确认 `git log --oneline -1` == 5f4c409。若基线存在已知失败，逐条记录并标注"先于本次改动已存在"，不得顺手修复（Non-goal 边界）。
  - 验收：分支存在且起点为 5f4c409；基线 pytest 结果已记录（全绿，或已知失败清单+区分说明）；工作区干净。
  - 依赖：无

- [ ] T1.2 锁定 1c19a63 移植基准并落档实证结论
  - 文件：无代码修改；产出为移植参考文本与实证记录（可暂存于本地临时文件或直接在审阅中读取，不入库）
  - 修改：`git show 1c19a63:xstars/plot_engine.py`、`git show 1c19a63:xstars/main.py`、`git show 1c19a63:tests/test_presets.py` 提取 1c 原版内容作为移植唯一基准。rev 2 已预实证（§4.6-2）：1c 原版 `_qpcr_bars` 含 `ax.set_xticks`/`ax.set_xticklabels`（L173-174），工作区与 1c 对 plot_engine/test_application_analysis 零差异。实施时在 feat 分支上复核一次并写入提交信息或 PR 描述备查。
  - 验收：1c 原版含 set_xticklabels 的实证结论已复核并落档（预期与 rev 2 一致）；移植一律以 1c 原版为准，不引入 1c 之后的任何 WPS 变更。
  - 依赖：T1.1

- [ ] T1.3 修正模板示例 TNF-a Control Ct（R12）
  - 文件：`XSTARS_Templates.xlsx`（修改；qPCR sheet Q11/Q12 两格）
  - 修改：TNF-a Control Ct `28.2`→`28.4`（Q11）、`27.8`→`27.7`（Q12），使 control ΔCt 方差 ≠ 0；不改动其他任何 sheet/单元格；不动本地未跟踪的 "XSTARS_Templates copy.xlsx"。写入前确认文件未被 Excel 进程占用。
  - 验收：用修改后模板数据重跑 `transform_labeled` + `StatsEngine`：TNF-a 新旧口径均为 ANOVA→Tukey（无 Dunn fallback，预验证 control fold-change 方差 0.0109）；IL-6 结果与 rev 2 预验证一致；工作簿其余区域未被改动。
  - 依赖：无

### M2 统计 gate 移植与 6 调用点接入

- [ ] T2.1 在 `xstars/presets/qpcr.py` 实现 qPCR-only log2 stats gate helper
  - 文件：`xstars/presets/qpcr.py`（修改；模块尾部新增模块级函数）
  - 修改：新增 `stats_input_frame(transformed: pd.DataFrame, preset: BasePreset | None) -> pd.DataFrame`——`isinstance(preset, QPCRPreset)` 时返回 `np.log2` 后按原 index/columns 重建的 DataFrame，否则原样返回（逻辑逐行对照 WPS application/analysis.py:234-251，含 docstring 意图：log2(2^-ΔΔCt) = -ΔΔCt 精确成立）；另增 config 变体 `stats_input_frame_for_config(transformed: pd.DataFrame, config: PrismConfig) -> pd.DataFrame`（谓词 `config.experiment_preset is ExperimentPreset.QPCR`，与 plot_engine `_is_qpcr` 同谓词；两谓词等价性由 `register_preset(ExperimentPreset.QPCR)`（qpcr.py:27）保证）。`PrismConfig` 以 TYPE_CHECKING 延迟导入避免循环依赖（如需要）。不加数值 guard（R6）。
  - 验收：非 qPCR preset/None 输入原样返回（同一对象或等值 frame）；qPCR 输入逐元素等于 log2；模块仍不 import xlwings（`python -c "import xstars.presets.qpcr"` 在无 xlwings 环境可执行）；T4.1 单测覆盖。
  - 依赖：T1.1

- [ ] T2.2 main.py 非 export 的 4 个 qPCR-capable 调用表达式接入 gate
  - 文件：`xstars/main.py`（修改）
  - 修改：在 `_run_preset_impl`（定义 :436，调用 :508-509）、`_run_qpcr_labeled`（定义 :637，调用 :667-668）、`_run_impl`（定义 :744，调用 :781-782）、`_run_quick_impl`（定义 :829，调用 :842-843）中，把传给 `StatsEngine(...).analyze(...)` 的 qPCR-可达实参从 transformed frame 换为 `stats_input_frame(transformed, preset)`（preset 实例在作用域时）或 `stats_input_frame_for_config(transformed, config)`（仅 config 在作用域时），并添加 helper import。**明确不动**：`_run_elisa_impl`（:330-331）与 `_run_wb_labeled`（:587-588）两调用点（R3）。labeled 路径对每个 target 的 fold-change frame 分别过 gate（对照 WPS `_analyze_labeled` :430-432 模式）。写回/绘图数据保持 transformed frame 不变（R8）。
  - 验收：静态核查 `git grep -n "stats_input_frame" xstars/main.py` 命中 6 处（含 T2.3 的 2 处）；`git grep -nE "np\.log2|log2\(" xstars/main.py` 零命中（log2 只在 helper 内）；ELISA/WB 调用点代码审阅无 gate。
  - 依赖：T2.1、T1.2

- [ ] T2.3 main.py export/include_stats 的 2 个调用表达式接入 gate
  - 文件：`xstars/main.py`（修改）
  - 修改：`_run_export_impl`（定义 :1066）中 labeled `target_dfs` 分支（:1477-1478）与普通 `df_wide` 分支（:1499-1500），统计实参分别过 gate（labeled 分支逐 target；两处均以 config 谓词变体或 preset 实例按作用域可用对象选择，语义对应 WPS re-render 的 `stats_input_frame` 调用模式）。export 写回内容不变（R8）。
  - 验收：与 T2.2 合并静态核查 6 处全命中；export 两分支代码审阅均经 gate 且仅 qPCR config 生效。
  - 依赖：T2.1、T1.2

- [ ] T2.4 qPCR 输出标签落地（R11）
  - 文件：`xstars/presets/qpcr.py`（新增常量与辅助函数）、`xstars/main.py`（修改 qPCR 写回调用点）
  - 修改：① presets/qpcr.py 新增 `PROCESSED_DATA_SUFFIX = " (2^-ΔΔCt)"`、`PVALUE_LABEL = "p-value(−ΔΔCt)"`（rev 4 定稿措辞）、`qpcr_stats_table(stats_df: pd.DataFrame) -> pd.DataFrame`（仅重命名 `p-value` 列，其余列不动）；② main.py 的 qPCR 写回路径（`_run_qpcr_labeled`、`_run_preset_impl`/`_run_impl`/`_run_quick_impl` 的 qPCR preset 分支、`_run_export_impl` 两分支的 qPCR config 情形）把 processed data 标题追加 `PROCESSED_DATA_SUFFIX`，stats 表写回前过 `qpcr_stats_table`。注意：main 版 main.py 的写回调用点行号未在 explore 报告中锚定（证据缺口），实施时在 feat 分支上读取 main 版文件定位，不得引用 wps 工作区行号。
  - 验收：`git grep -n "2^-ΔΔCt"` 仅命中 qPCR 相关路径与 presets/qpcr.py；ELISA（:330-331 所在函数）与 WB（:587-588 所在函数）的标题与 `p-value` 列名代码审阅不变；T4.1 ④ 单测通过。
  - 依赖：T2.1、T1.2

### M3 plot_engine qPCR 图形移植

- [ ] T3.1 移植 `_is_qpcr`/`_qpcr_geo_stats`/`_log_error_value` 到 main PlotEngine
  - 文件：`xstars/plot_engine.py`（修改；helpers 区，参照 main 版 `_error_value` :336-352 所在区域尾部插入）
  - 修改：按 main 的 PlotEngine 结构移植三个方法，逻辑逐行对照 WPS plot_engine.py:418-456（`_is_qpcr` :418-420 config 谓词；`_qpcr_geo_stats` :422-438 几何均值 + 上下不对称偏移；`_log_error_value` :440-456 SEM/SD/CI95，scipy t 局部导入）。确保 main 版 plot_engine 的 `from .config import ...` 含 `ExperimentPreset`（如缺则补入 import 列表）。不加数值 guard（R6）。
  - 验收：三个方法存在且行为与 WPS 版逐值一致（T4.2 断言 `2^(mean log2)` 几何均值与 `2^(mean±err)` 不对称端点）；非 qPCR 路径不受影响。
  - 依赖：T1.1、T1.2

- [ ] T3.2 移植 `_qpcr_bars` 并在 `_bar_scatter` 加 qPCR 分发
  - 文件：`xstars/plot_engine.py`（修改；`_bar_scatter` main 版 :79-118 为插入点）
  - 修改：在 main `_bar_scatter` 内、seaborn barplot 之前插入 `if self._is_qpcr(): self._qpcr_bars(ax, df_wide, groups) else:` 包住原 seaborn 路径，`show_points` 的 stripplot 保持两分支共用（分发结构对照 WPS plot_engine.py:85-127）；`_qpcr_bars` 本体按 1c 原版移植（rev 2 实证：1c 含 `ax.set_xticks`/`ax.set_xticklabels`（L173-174），照搬保留，R4；WPS 工作区版本 :129-174 与 1c 零差异可作参照）。仅特化 BAR_SCATTER，violin/line 不动。
  - 验收：qPCR + BAR_SCATTER 走 `_qpcr_bars`（柱高=几何均值、误差条不对称）；非 qPCR BAR_SCATTER 输出与移植前一致（既有 TestBarScatter/TestErrorBars 全绿）；显著性与轴标注由 `plot()` 统一后处理不与新 bar 坐标冲突（人工审阅 + M5 记录）。
  - 依赖：T3.1、T1.2

### M4 测试移植

- [ ] T4.1 扩展 `tests/test_presets.py`：gate 单测 + log-space p 值 integration
  - 文件：`tests/test_presets.py`（修改；追加测试类/方法，不新建文件）
  - 修改：① gate 单测：非 qPCR preset（如 WBPreset）与 None 原样返回；QPCRPreset 输入逐元素 log2；config 变体谓词等价。② log-space p 值 integration（plain）：构造 ΔCt 输入（可移植 WPS test_presets.py :507-531 的 rng 数据模式 + QPCRStatsSpaceTests :589-617 的三组 ΔCt 手工数据意图），`QPCRPreset.transform` 得 fold change，过 `stats_input_frame` 喂 `StatsEngine`，断言 decision_path 含 ANOVA 且 p 值与 `scipy.stats.tukey_hsd(log2fc[:,i], ...)` 手算对拍（delta≤1e-12，对照 :617-626 模式）。③ labeled 对拍：移植 QPCRStatsSpaceTests :619-647 意图（GAPDH reference 三组数据，逐 target Tukey 对拍），用 DataFrame 直接构造，**不引入** SelectionPayload 等 WPS 依赖。~~**不移植** :804-826 tick-label 测试~~ **[superseded rev 2]**：该测试属 1c 自身内容，其意图改由 T4.2 移植到 `tests/test_plot_engine.py`（R4 修订版）。④ 输出标签（R11）：`qpcr_stats_table()` 仅重命名 `p-value` 列为 `PVALUE_LABEL`、其余列不动；`PROCESSED_DATA_SUFFIX` 常量值正确。
  - 验收：新增测试全部通过；`tests/test_application_analysis.py` 未创建；断言包含 transformed frame 全正（fold-change 不变量）、p 值数值对拍、qPCR stats 表列名重命名（含非 qPCR frame 不受影响的对照）。
  - 依赖：T2.1、T2.2、T2.3

- [ ] T4.2 扩展 `tests/test_plot_engine.py`：几何均值/不对称误差条测试
  - 文件：`tests/test_plot_engine.py`（修改；追加测试类，沿用既有 Agg/matplotlib 导入模式）
  - 修改：移植 QPCRStatsSpaceTests `test_qpcr_geo_stats_derive_from_log_space`（:649-666）意图：`PrismConfig(experiment_preset=QPCR)` 下 `_qpcr_geo_stats(Series([1.0, 2.0, 4.0]))` 断言 geo=2.0、lower/upper 等于 `2^(1∓sem_log)`/`2^(1±sem_log)` 且 upper>lower；另加 qPCR BAR_SCATTER 冒烟（`PlotEngine(config).plot(df_wide)` 成功出图、axes 存在）与非 qPCR config 走原路径的对照断言；再加 tick-label 回归（rev 2 新增，移植 1c :804-826 意图）：`show_points=False` 下 qPCR BAR_SCATTER 的 `ax.get_xticklabels()` 等于组名（对应 1c `_qpcr_bars` L173-174 行为）。不依赖 xlwings。
  - 验收：新增测试通过；既有 TestBarScatter/TestErrorBars/TestOptions（main :15-81）不修改且全绿；tick-label 断言通过（组名刻度）。
  - 依赖：T3.1、T3.2

### M5 全量验证与 Draft PR

- [ ] T5.1 全量验证与静态核查
  - 文件：无新改动（验证动作；如核查发现问题，回对应 T 任务修复后重跑）
  - 修改：执行 §8 Validation contract 全部可执行项（pytest 全量、gate 覆盖 grep、非 qPCR 不变核查、图形分发核查、wps 分支零改动核查），逐项记录结果。
  - 验收：§8 第 1-6 项全部通过；任何失败项回溯修复并复跑至全过。
  - 依赖：T4.1、T4.2

- [ ] T5.2 创建 Draft PR
  - 文件：无（git/平台操作）
  - 修改：推送 `feat/qpcr-log-space-excel-sync`，按 §9 Git 策略开 Draft PR（base=main，标题与描述草稿见 §9），标记 Draft。
  - 验收：Draft PR 存在、base=main、diff 仅涉 §9 文件级范围所列文件。
  - 依赖：T5.1

## 8. Validation contract

| # | 检查项 | 命令/验证方式 | 预期结果 | 通过标准 |
| --- | --- | --- | --- | --- |
| 1 | main 基线绿 | 分支起点运行 `python -m pytest tests -v` | 基线全绿（或已知失败已记录并区分，见 T1.1） | 0 failed（或已知失败清单与本次改动无因果） |
| 2 | pytest 全量（**验收标准，R7**） | feat 分支上 `python -m pytest tests -v` | 全部通过，含 T4.1/T4.2 新增测试 | 0 failed、0 error |
| 3 | gate 覆盖静态核查（访谈确认以静态核查替代逐入口 mock） | `git grep -n "stats_input_frame" xstars/main.py`；`git grep -nE "np\.log2\|log2\(" xstars/main.py`；逐点代码审阅 :508-509、:667-668、:781-782、:842-843、:1477-1478、:1499-1500 | 6 个调用表达式均经 gate；main.py 内无散落 log2（log2 仅在 presets/qpcr.py helper 与 plot_engine helpers 内） | 6 处命中、0 散落、审阅清单逐点签字 |
| 4 | ELISA/WB 回归不变核查 | 代码审阅 main.py :330-331（ELISA）、:587-588（WB labeled）无 gate；`git diff main...HEAD -- xstars/stats_engine.py tests/test_stats_engine.py` 为空且该测试全绿 | 非 qPCR 调用点与引擎零改动 | 两调用点无 gate + 引擎文件 diff 为空 + test_stats_engine.py 全绿 |
| 5 | 图形分发核查 | `git grep -nE "_qpcr_bars\|_is_qpcr\|_qpcr_geo_stats\|_log_error_value" xstars/plot_engine.py`；既有 TestBarScatter/TestErrorBars 全绿；对照 `git show 1c19a63:xstars/plot_engine.py` 确认 `_qpcr_bars` 严格复刻（rev 2 实证：1c 含 set_xticklabels，移植版须含 L173-174 对应行） | 4 个符号存在、分发生效、非 qPCR 路径输出不变、1c 复刻无增删（含 tick-label 行为） | grep 命中 + 既有测试全绿 + 复刻对照通过 + tick-label 测试通过 |
| 6 | wps 分支零改动核查 | `git log --oneline main..HEAD`（仅本次移植提交）；`git diff --stat main...HEAD`（仅 §9 所列 5 文件）；`feature/wps-support` ref 无新提交（`git rev-parse feature/wps-support` 不变） | 无 WPS 范围文件（application/、contracts、export.py、test_application_analysis.py）流入；wps 分支原样 | 提交清单与 diff 范围核查通过 |
| 7 | 真实 Excel 人工验证（**后置，非本 Plan 验收项，R7**） | 合并后在真实 Excel 中人工验证：普通 qPCR、labeled qPCR、Quick、export include-stats，各含 SEM/SD/CI95 三种 error-bar 配置；同时核对 processed data 标题与 p-value 列名（R11） | 统计表 p 值与 log 空间一致；qPCR 柱图为几何均值+不对称误差条；ELISA/WB 输出与修复前一致 | **不适用**（后置动作）；责任人：用户；结果记录于后续 issue/PR 评论 |
| 8 | 模板数据核查（R12/T1.3） | 修改后用模板 qPCR sheet 数据重跑 `transform_labeled`+`StatsEngine`（含新旧两种口径） | TNF-a 新旧口径均为 ANOVA→Tukey，无 Dunn fallback；IL-6 与 rev 2 预验证一致；其余 sheet 未变 | 两个 gene 的 decision_path 与预验证一致 |
| 9 | qPCR 输出标签核查（R11/T2.4） | `git grep -n "2^-ΔΔCt\|p-value(−ΔΔCt)"`；ELISA/WB 写回路径代码审阅；T4.1 ④ 单测 | 标签仅出现在 qPCR 路径；ELISA/WB 标题与 `p-value` 列名不变 | grep 命中范围符合 + 审阅通过 + 单测通过 |

注：第 3 项以静态核查（grep + 代码审阅）作为入口覆盖的验证方式，系访谈确认（xlwings 宏入口 mock 测试成本过高，验收以 pytest 全量为准）；第 7 项不能在本 Plan 内执行，原因=需真实 Excel 宿主与人工操作，责任人=用户，已按访谈后置。

## 9. 文件级修改范围 + 风险 / 回滚 / 待决事项 + Git 策略

### 9.1 文件级修改范围

| 文件 | 操作 | 说明 |
| --- | --- | --- |
| `xstars/presets/qpcr.py` | **修改** | 新增 `stats_input_frame` + `stats_input_frame_for_config` helper（§4.5 落点决策；不新建文件）；新增输出标签常量 `PROCESSED_DATA_SUFFIX`/`PVALUE_LABEL` 与 `qpcr_stats_table()`（R11） |
| `XSTARS_Templates.xlsx` | **修改** | qPCR sheet Q11/Q12 两格（TNF-a Control Ct，R12/T1.3）；其余内容不动 |
| `xstars/main.py` | **修改** | 6 个 qPCR-capable 调用表达式接入 gate（T2.2/T2.3）+ helper import；ELISA :330-331、WB :587-588 不动 |
| `xstars/plot_engine.py` | **修改** | `_bar_scatter`（:79-118）加 qPCR 分发；新增 `_qpcr_bars`/`_is_qpcr`/`_qpcr_geo_stats`/`_log_error_value`；按需补 `ExperimentPreset` import |
| `tests/test_presets.py` | **修改** | 追加 gate 单测 + log-space p 值 integration（plain + labeled Tukey 对拍） |
| `tests/test_plot_engine.py` | **修改** | 追加几何均值/不对称误差条测试 + qPCR BAR_SCATTER 冒烟 |
| `xstars/stats_engine.py` | **明确不修改** | 引擎不感知 preset（R10） |
| `xstars/application/**`（含 export.py） | **明确不修改/不创建** | 路径 A 边界（R1）；main 不存在该目录，不得流入 |
| `tests/test_application_analysis.py` | **明确不创建** | R5；不复制 WPS application 测试 |
| `tests/test_stats_engine.py` | **明确不修改** | 通用引擎测试零改动（§8 第 4 项核查项） |
| `xstars/config.py`、`xstars/styles.py`、其余 presets、installer/文档 | **明确不修改** | 无关联改动面 |
| `plans/20260903-qpcr-log-space-excel-sync.md`、`plans/explore-20260903-…md` | 不修改（本 Plan 落盘后） | 规划产物 |

### 9.2 风险

| ID | 风险 | 等级 | 触发条件 | 影响 | 缓解 |
| --- | --- | --- | --- | --- | --- |
| R1 | 移植遗漏 qPCR-capable 调用点（尤其 export 两分支） | 高 | 6 表达式分散于 5 函数；手工逐点接入 | 同一数据不同入口不同口径，双轨状态残留 | §8 第 3 项静态核查（6 处 grep 命中 + 逐点审阅清单）；T2.2/T2.3 按表达式粒度拆任务 |
| R2 | 冲突处理/照搬引入 WPS 范围（application/contracts/export 流入） | 中 | 试图用 cherry-pick 结果直接落盘，或从工作区版本复制而非 1c 原版 | 违反 R1 路径边界，review 面与回归面扩大 | 手工移植为唯一路径；§8 第 6 项 diff 范围核查（仅 5 文件）；T1.2 锁定 1c 基准 |
| R3 | 图形显著性与轴标注与新 bar 坐标兼容性 | 中 | `_qpcr_bars` 的 ax.bar/errorbar 与 `plot()` 统一 annotation 层的交互在 main 结构下与 WPS 有差异 | 标注错位或遮挡 | WPS 同分发结构已验证（1c + 测试）；T4.2 冒烟 + §8 第 5 项；真实 Excel 人工后置验证覆盖此项 |
| R4 | 两分支临时重复逻辑 | 中 | main 的 presets/qpcr.py helper 与 WPS application/analysis.py `_stats_input_frame` 并存 | PR #1 合并时同域冲突/双实现漂移 | 命名对齐（§4.5-4）便于识别收敛；PR 描述显式声明与 PR #1 的关系与去重约定（PR #1 侧解决冲突） |
| R5 | 极端数值边界（0/inf/非有限 → log2 产生 -inf/NaN 进 StatsEngine） | 低 | qPCR transform 指数 underflow 至 0 或非有限输入 | warning/-inf 进入决策树，结果异常 | 访谈决策 6：沿用 1c 行为不新增 guard（正常有限 ΔΔCt 输出严格为正，explore 报告 §3.4）；残余风险已记录，如需 guard 另开需求 |
| R6 | ~~`_qpcr_bars` 基准歧义导致误移植 post-1c 修复~~【rev 2 已消除】 | 低（已消除） | rev 1 假设工作区 ≠ 1c；实证（§4.6-2）证明工作区与 1c 对相关文件零差异、tick-label 修复属 1c 自身 | 无残余影响；T1.2 保留为落档确认 | T1.2 复核并落档；§8 第 5 项含复刻对照检查 |
| R7 | main 基线本身存在已知失败干扰验收判定 | 低 | main @ 5f4c409 的 pytest 非全绿 | 无法区分新失败与存量失败 | T1.1 先记录基线结果并区分；验收只要求"相对基线无新增失败且新增测试全过"（若基线全绿则为 0 failed） |
| R8 | 标签两分支漂移：Excel 版新增 R11 标签，WPS 版（application/analysis.py）暂无同名标签 | 中 | PR 合并后两侧输出标签不一致，直到 WPS 侧对齐 | 同一数据两宿主输出标签不同 | 已显式列为后置对齐动作（§9.4）；PR 描述注明；WPS 对齐归属 PR #1 侧或后续小 PR |
| R9 | 模板文件被 Excel 占用导致写入失败 | 低 | 用户本机 Excel 打开着模板或其副本（存在 `~$` 锁文件） | openpyxl 保存报 Permission denied | T1.3 写入前检测占用并提示关闭；失败则重试，不阻塞其他任务 |

### 9.3 回滚

- **回滚步骤**：合并前随时可回滚——`git switch main && git branch -D feat/qpcr-log-space-excel-sync`；若 Draft PR 已开，关闭 PR 即可。main 与 feature/wps-support 在整个过程中零改动，无需其他恢复动作。
- **数据兼容性**：无数据/配置/持久化格式迁移——本变更只影响统计计算空间与图形渲染口径；Excel 工作簿、PrismConfig 字段、preset 输入输出格式均不变（R8：写回/绘图数据仍是 fold change）。回滚后用户可见行为即恢复 main 现状，无残留状态。

### 9.4 待决事项

- **无未决产品/范围决策**（访谈 2026-09-03 已清空：路径、等价范围、入口覆盖、MINOR 处置、测试落点、数值边界、验收标准全部确定，见 R1-R7）。
- 遗留**事实确认项**（非决策，不需用户批复）：~~`_qpcr_bars` 的 1c 原版是否含 set_xticklabels~~【rev 2 已实证：1c 含该行为（L173-174），工作区与 1c 零差异，"Fix 3c" 属 1c 自身内容；见 §4.6-2】。T1.2 仅为落档复核。
- 遗留**后置动作**（已按访谈/用户指令记录，非本 Plan 范围）：真实 Excel 人工验证（§8 第 7 项）；PR #1 合并时的两分支 gate 去重（R4 风险区）；**R11 标签与 R12 模板数据的 WPS 侧对齐**（application/analysis.py 标题与列名 + 模板文件，归属 PR #1 侧或后续小 PR，R8 风险区）。
- **rev 3 新增范围说明**：R11/R12 系用户 2026-09-03 追加指令（在路径 A 已获认可的前提下提出），非 Agent 自行扩张；除 R11/R12 外未新增其他范围，"严格复刻 1c"决策继续适用于统计/图形行为本身，标签为输出层叠加、不影响统计行为。

### 9.5 Git 策略

- **分支名**：`feat/qpcr-log-space-excel-sync`，自 `main` @ `5f4c4099bd7c578e8d3fde9e59de0ef77a4dc3b9` 创建。
- **PR 形态**：单一 **Draft PR**，base = `main`。
- **Draft PR 标题**：`fix(qpcr): run hypothesis tests on the linear −ΔΔCt/log2FC space (Excel port of 1c19a63)`
- **PR 描述草稿**（可直接使用）：

  > ## Background
  >
  > Commit `1c19a63` on `feature/wps-support` moved qPCR hypothesis tests to the linear log2 fold-change (−ΔΔCt) space and switched qPCR BAR_SCATTER charts to geometric means with asymmetric log-space SEM/SD/CI95 error bars. That fix currently only reaches the WPS host; the Excel/main line still runs the generic decision tree and bar charts directly on the nonlinear `2^-ΔΔCt` ratio scale, so the same data yields different statistics/figures depending on the entry point.
  >
  > This PR ports the behavioral fix to main **without** importing the WPS application layer, contracts, or `application/export.py` (agreed scope: manual port on a fresh branch from main).
  >
  > ## Changes
  >
  > - `xstars/presets/qpcr.py`: new `stats_input_frame` / `stats_input_frame_for_config` helpers — qPCR-only `np.log2` gate (`log2(2^-ΔΔCt) = −ΔΔCt` holds exactly); non-qPCR frames pass through unchanged. Module stays xlwings-free so pytest can import it headless.
  > - `xstars/main.py`: all 6 qPCR-capable `StatsEngine` call sites now route through the gate — `_run_preset_impl` (:508-509), `_run_qpcr_labeled` (:667-668), `_run_impl` (:781-782), `_run_quick_impl` (:842-843), and both `_run_export_impl` branches (:1477-1478, :1499-1500). ELISA (:330-331) and WB-labeled (:587-588) are intentionally untouched. Writeback/plot data remains fold change.
  > - `xstars/plot_engine.py`: ports `_is_qpcr`, `_qpcr_geo_stats`, `_log_error_value`, `_qpcr_bars` and the BAR_SCATTER dispatch, replicating the `1c19a63` version verbatim (including its known MINOR state — the `_qpcr_bars` xticklabels issue is explicitly out of scope here).
  > - `tests/test_presets.py`: gate unit tests + log-space p-value integration with hand-computed `scipy.stats.tukey_hsd` cross-checks (plain + labeled).
  > - `tests/test_plot_engine.py`: geometric-mean / asymmetric error-bar assertions (ported from the WPS `_qpcr_geo_stats` test intent).
  >
  > ## Relation to PR #1 / #3 (feature/wps-support)
  >
  > Independent of and unordered relative to the WPS PR(s): either may merge first. This PR intentionally duplicates the gate logic that lives in `xstars/application/analysis.py` on the WPS branch (helpers are name-aligned to ease later convergence). If/when same-domain conflicts arise, they are resolved on the PR #1 side, where the application layer becomes the single home of this logic.
  >
  > ## Validation
  >
  > - `python -m pytest tests -v` — full suite green (this is the acceptance bar agreed with the user).
  > - Static audit: all 6 qPCR-capable call expressions route through `stats_input_frame` (grep + per-site review); ELISA/WB call sites and `xstars/stats_engine.py` unchanged.
  > - Manual verification in real Excel (plain/labeled qPCR, Quick, export include-stats; SEM/SD/CI95) is deferred and tracked as a follow-up, not part of this PR's acceptance.

- **PR 拆分决策**：**单 PR**。依据：5 个 milestone 是同一行为修正的连续切片（gate→调用点→图形→测试→验证），不存在可独立交付/独立验收的子系统；中途拆分会产生"统计已改、图未改"或"实现已改、测试未跟"的不可验收中间态；总 diff 预计 5 文件、规模可控。
- **合并顺序**：本 PR 与 PR #1（feature/wps-support）**互相独立、先后不限**。若本 PR 先合并：PR #1 合并时如产生同域冲突（`xstars/main.py`、`xstars/plot_engine.py`、`tests/test_presets.py` 及 gate 重复逻辑），**在 PR #1 侧解决**——application 层成为该逻辑的唯一归属，收敛/删除本 PR 引入的 `presets/qpcr.py` helper 形态。若 PR #1 先合并：本 PR 需 rebase 并按同一原则在 PR #1 侧（即以 application 层为准）去重后重提。

---

## 10. 完成前自查（契约核对结果）

- [x] 9 段结构齐全且顺序固定（Goal → Requirements → Non-goals → Research summary → Gap analysis → Milestone 表格 → 分 milestone To-do → Validation contract → 文件级修改范围+风险/回滚/待决/Git 策略）。
- [x] Milestone 数量 = 5（≤10，符合目标 ≤7）。
- [x] 每个功能缺口至少映射一个 To-do 任务 ID：G1→T2.1/T2.2/T2.3；G2→T2.2/T2.3/T5.1；G3→T2.1/T4.1；G4→T3.1/T3.2；G5→T4.1/T4.2；G6→T1.1/T1.2。**无孤立缺口**；反向核查：所有任务均有对应缺口或横切约束支撑（T1.x→G6/R1/R7，T5.x→G2/R7）。
- [x] 每个任务都有文件、修改、验收、依赖四要素。
- [x] 所有歧义进入待决事项/证据缺口：访谈已清空产品决策（§9.4）；rev 2 以实证解决了 rev 1 的唯一事实确认项（`_qpcr_bars` 1c 版本，§4.6-2/T1.2，未改任何产品决策——"严格复刻 1c"含义随实证自然确定）；后置动作（真实 Excel 验证、PR #1 去重）均已显式记录，未擅自拍板任何产品行为。
- [x] 输出仅写入合规规划路径 `plans/20260903-qpcr-log-space-excel-sync.md`，未创建或覆盖任何源码、测试、配置、脚本或生成物。
- [x] main 侧行号锚点全部复用 explore 报告已验证锚点，未虚构任何 main 侧行号；WPS 侧锚点来自工作区实读与 explore 报告。

## Changelog

| Rev | 日期 | 变更 | 依据 |
| --- | --- | --- | --- |
| 1 | 2026-09-03 | 初版 | explore 报告（plans/explore-20260903-qpcr-log-space-excel-sync.md）+ 访谈 2026-09-03 |
| 2 | 2026-09-03 | 实证修正：1c19a63 原版 `_qpcr_bars` 已含 set_xticklabels（L173-174），"Fix 3c" 测试 :804-826 属 1c 自身内容；rev 1 的 MINOR 前提失效，R4/G6/T1.2/T3.2/T4.1/T4.2/M1/M3/§8-5/R6/§9.4 相应修订（superseded 标注保留） | 父 Agent 实证：`git show 1c19a63:xstars/plot_engine.py`、`git diff 1c19a63 -- xstars/plot_engine.py tests/test_application_analysis.py`（均空）、`git log 1c19a63..feature/wps-support -- xstars/plot_engine.py`（空） |
| 3 | 2026-09-03 | 新增用户追加范围：R11 qPCR 输出标签（`Processed Data (2^-ΔΔCt)` / `p-value (calculated on ΔCt)`，仅 qPCR 路径）；R12 模板 TNF-a Control Ct 改为 [28, 28.4, 27.7]（Q11/Q12 两格，避免 Dunn fallback）。新增 T1.3/T2.4、T4.1-④、§8-8/9、风险 R8/R9、文件级范围 +XSTARS_Templates.xlsx；非目标第 3 条同步 superseded（rev 2 遗留） | 用户 2026-09-03 追加指令；预验证：候选 A 使 TNF-a control fold-change 方差=0.0109，新旧口径均走 ANOVA→Tukey，新口径 LPS vs LPS+Dex p=0.0435 |
| 4 | 2026-09-03 | p-value 标签定稿为 `p-value(−ΔΔCt)`（用户指定，替换 rev 3 的 `p-value (calculated on ΔCt)`）；状态改"已批准" | 用户批准 rev 3 并指定唯一修订（2026-09-03） |
