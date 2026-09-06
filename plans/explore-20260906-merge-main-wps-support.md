# Explore 报告：合并 main 与 feature/wps-support 分支

- 日期：2026-09-06
- 主题：`main` 与 `origin/feature/wps-support` 的分支合并
- 性质：纯调研，未修改任何项目文件（`/tmp` 下的提取文件为临时对比副本）
- 方法说明：子代理基础设施（codebase-cartographer）在本环境不可用（前台子代理无法加载扩展工具、后台子会话无法创建），本次调研由主编排器直接执行，全部结论附 git 命令或路径+行号证据。

## 1. Implementation map

### 1.1 分支拓扑（已验证）

```text
            5f4c409 (merge-base, "Add demo video link to README")
           /   |    \
        main   feat/macos-support   feature/wps-support
        +12    +11 (含 macos 工作)   +53 (WPS 全量, +20254 行)
```

- `git merge-tree --write-tree main origin/feature/wps-support` 模拟合并：**恰好 4 个冲突文件**：`XSTARS_Templates.xlsx`（二进制）、`tests/test_presets.py`、`xstars/main.py`（11 处冲突块）、`xstars/plot_engine.py`。
- 双方同时修改的文件仅这 4 个（`comm -12` 对 `git diff --name-only` 两侧清单求交验证）；`wps-addon/`、`xstars/application/`、`xstars/wps_service.py` 等约 20k 行为 wps 侧独有新增，合并时无冲突直接并入。
- GitHub 现状：PR #1 `feature/wps-support → main` 为 open Draft；PR #2 `feat/macos-support → main` 为 open Draft（与本任务独立的兄弟分支，当前工作区即检出该分支，工作区版本 ≠ main）。

### 1.2 同一功能的两种实现（冲突根源）

双方共享提交 `1c19a63`（PR #3，"run hypothesis tests on the linear -dCT/log2FC space"），之后：

- **main 侧（PR #4，已合并）**：把 1c19a63 的 qPCR log-space 工作**手工移植**回 main 的内联架构：逻辑落在 `xstars/presets/qpcr.py:168-204`（`_log2_stats_frame` / `stats_input_frame` / `stats_input_frame_for_config` / `qpcr_stats_table`）+ `xstars/main.py`（+274/-94，stats gate 调用点与输出标签）+ `xstars/plot_engine.py:129-186`（`_qpcr_bars` 几何均值柱状图，e5b48e1 "port from 1c19a63"）。
- **wps 侧**：把宿主编排逻辑**抽取**到宿主无关应用层（db5775b，M2）：`xstars/application/analysis.py`（932 行，含同一 log-space gate：`_stats_input_frame` :231-253、公共 `stats_input_frame` :255-268、`PROCESSED_DATA_SUFFIX`/`PVALUE_LABEL` :265-267、`qpcr_stats_table` :270-276）+ `contracts.py`（510 行）+ `export.py`（372 行）+ `worker.py`（761 行）；`xstars/main.py` 缩为宿主适配器（+141/-276）。
- wps 侧 `274e169`（2026-09-03 20:23，晚于 main 模板修复 eaa9f10 的 18:17）**显式将标签/模板/绘图对齐 PR #4**：提交信息确认模板 TNF-a Control Ct R11/R12=[28, 28.4, 27.7]、plot_engine C408 对齐 main、analysis.py 增加 `qpcr_stats_table` 重命名助手。

### 1.3 数据流（合并后目标形态）

`main.py`（宿主适配：选择区读取、宿主对话框、写回执行）→ `application/analysis.py::analyze_dataframe`（清洗→preset 变换→log-space gate→StatsEngine→绘图数据）→ `application/export.py` / `worker.py`（写回计划、多图工件）→ `PlotEngine`（柱状/折线等 ChartType，qPCR 走 `_qpcr_bars`/`_line` 特化）。stats 引擎 `xstars/stats_engine.py` 不感知 preset（两侧一致未改）。

## 2. Key files and symbols

| 文件 | 符号/节点 | 锚点 | 关系 |
| --- | --- | --- | --- |
| `xstars/main.py` (main 侧) | `run_qpcr`、`_run_qpcr_labeled`、stats gate 调用点 | /tmp/xstars-main-side/main.py:505,679；conflict #4/#6/#7 @508/844/924、#5/#10/#11 @721/1691/1723 | main 内联 qPCR 编排，合并时被 wps 应用层调用取代 |
| `xstars/presets/qpcr.py` (main 侧) | `_log2_stats_frame`、`stats_input_frame`、`stats_input_frame_for_config`、`qpcr_stats_table` | main:xstars/presets/qpcr.py:168-204 | PR #4 的 log-space gate 本体（wps 侧无此文件改动，无冲突；语义已被 wps application 层复刻） |
| `xstars/application/analysis.py` (wps 侧) | `_stats_input_frame`、`stats_input_frame`、`PROCESSED_DATA_SUFFIX`、`PVALUE_LABEL`、`qpcr_stats_table`、`analyze_dataframe` | origin/feature/wps-support:xstars/application/analysis.py:231-276, 278+ | wps 应用层：与 PR #4 语义对齐的 gate/标签实现（274e169） |
| `xstars/application/contracts.py / export.py / worker.py` (wps 侧) | 写回契约/导出/多图工件 | origin/feature/wps-support:xstars/application/{contracts,export,worker}.py（510/372/761 行） | wps 独有新增，无冲突 |
| `xstars/plot_engine.py` | `_qpcr_bars` :129-186、`_is_qpcr` :409/418、`_qpcr_geo_stats` :413/422、`_line` gate | main 与 wps 侧对照 | 两侧已收敛，唯一语义差异 = wps `_line`:225-233 的 qPCR 几何均值 gate（main 无） |
| `tests/test_presets.py` | log-space 测试族 | main:tests/test_presets.py:539-715（`test_plain_qpcr_pvalues_match_manual_log2_space` 等） | main 侧 +308 行为超集；wps 侧 delta（+17/-5）= import 重排 + `analyze(log2_df)` 断言，已被 main 覆盖 |
| `tests/test_plot_engine.py` | qPCR 绘图回归 | main 侧 +77 行（wps 未改） | 无冲突，取 main |
| `XSTARS_Templates.xlsx` | 二进制模板 | main=eaa9f10 产物；wps=274e169 产物 | 两版均含 27.7/28.4 Ct 修复值（unzip 对比 sheet3 验证）；wps 版为后出的对齐版 |
| `plans/20260903-qpcr-log-space-excel-sync.md` | PR #4 计划文档 | main:plans/20260903-qpcr-log-space-excel-sync.md:53（Non-goal 10） | 声称"1c 只特化 BAR_SCATTER"，与 1c19a63 实际代码不符（见 Gap G2） |
| `wps-addon/`（wps 侧） | 官方 WPS 加载项前端（JS） | origin/feature/wps-support:wps-addon/*（含 tests/*.cjs） | 纯新增，无冲突；有独立测试套件 |

## 3. Existing patterns and conventions

- **Plan 文档**：`plans/<yyyymmdd>-<slug>.md`，9 段结构（如 `plans/20260903-qpcr-log-space-excel-sync.md`、`plans/20260829-macos-support.md`）。
- **提交信息**：Conventional Commits + 里程碑标签，如 `feat(qpcr): ... (T2.1-T2.4, M2)`、`docs(plan): close M6 ...`。
- **风格收敛**：pi-lens ruff 自动收敛提交（`a61c24b`、`6fef8f7`、`93b1741`），标注 behavior-neutral [skip-changelog]；main 侧 plot_engine 含 `cast(Any, ...)`（a61c24b），wps 侧无。
- **测试**：pytest（`tests/test_*.py`），wps-addon 另有 node test（`wps-addon/tests/*.cjs`）；测试断言风格含 `np.log2` 手工对数空间交叉验证（main:tests/test_presets.py:590-595）。
- **宿主无关性**：wps 侧 main.py 通过 `import_module("ttkbootstrap")` 动态导入（/tmp/xstars-wps-side/main.py:12 区域，conflict #2），宿主 I/O 全部走 application 层契约。

## 4. Candidate implementation paths

### 路径 A（推荐候选）：从 main 新建集成分支，merge `origin/feature/wps-support`

- 入口点：`git switch -c feat/merge-wps-support main && git merge origin/feature/wps-support`。
- 影响面：4 个冲突文件手工解决（预计取 wps 结构为主）；其余 +20k 行自动并入。
- 代价：中。需在 wps 架构上核对 PR #4 行为等价（tests 已双覆盖，验证成本低）。
- 兼容性：保留双方完整历史；PR #1（Draft）可选择在合并后 rebase/更新或直接关闭由集成分支 PR 取代。
- 适用条件：希望得到"main + WPS"的统一主线，且不动 wps 分支自身历史。

### 路径 B：在 `feature/wps-support` 分支上 merge main（更新现有 Draft PR #1）

- 入口点：`git switch feature/wps-support && git merge main`。
- 影响面：同样的 4 文件冲突，但解决结果直接落在 PR #1 分支。
- 代价：中低（少一个新 PR），但 PR #1 的 53 个提交审阅面已被 Plan 阶段前的多轮 review 消化过。
- 兼容性：main 历史并入 wps 分支；Draft PR #1 顺带完成"与 main 同步"。
- 适用条件：用户希望维持既有 PR #1 作为唯一合并载体。

### 路径 C（不推荐）：rebase wps 53 个提交到 main

- 代价：极高（53 提交重放、共享提交 1c19a63 会产生大量重复/冲突），破坏已 review 的 Draft PR 历史。
- 仅当用户明确要求线性历史才考虑。

### 冲突解决语义（适用 A/B）

- `xstars/main.py`：11 处冲突中 #4/#6/#7（内联流程 vs 应用层调用）、#5/#10/#11（stats gate 调用）取 **wps 侧**（274e169 已对齐 PR #4 语义）；#1/#2/#3/#8/#9（imports、ttkbootstrap 动态导入、matplotlib Agg、`Any` 注解）按 wps 宿主无关架构取 wps，逐处核对 main 侧仍被引用的 import。
- `xstars/plot_engine.py`：取 main 的 `cast(Any, ...)` 风格 + wps 的 `_line` qPCR gate（G2 产品决策待确认）。
- `tests/test_presets.py`：取 **main 侧**（超集，wps delta 已覆盖）。
- `XSTARS_Templates.xlsx`：取 **wps 侧**（274e169 对齐版，含 main 修复值，unzip 实证）。

## 5. Preliminary gap analysis

- **G1 结构冲突**：main 内联 qPCR 编排 vs wps application 层。现状：11 处冲突块（证据：merge-tree 冲突清单）。影响：main.py 无法自动合并。
- **G2 `_line` 行为分歧（真正语义分歧）**：wps/1c19a63 的折线图对 qPCR 使用几何均值+log 空间误差棒（origin/feature/wps-support:xstars/plot_engine.py:225-233）；main 的 PR #4 移植未包含（main:xstars/plot_engine.py:205-220 无 gate），且其 Plan Non-goal 10（main:plans/20260903-qpcr-log-space-excel-sync.md:53）基于"1c 只特化 BAR_SCATTER"的错误前提——1c19a63 实际代码含 `_line` gate（git show 1c19a63:xstars/plot_engine.py:225）。影响：合并后 qPCR 折线图跨宿主行为不一致（Excel 线=算术均值，WPS 线=几何均值）。需要产品决策。
- **G3 测试归属**：`tests/test_presets.py` main 侧超集 vs wps 侧小 delta。影响：需确认 wps 的 `analyze(log2_df)` 集成断言在 main 版本中有等价覆盖（初判已覆盖：main:tests/test_presets.py:606-647）。
- **G4 模板二进制**：无法三方合并，需择一。现状：wps 版为对齐版超集（274e169 提交信息 + unzip sheet3 数值对比：两版均含 27.7/28.4）。
- **G5 合并后回归验证**：wps 侧引入 274e169 之后的提交（93b1741 等 style）与 main 侧 ruff 收敛（a61c24b）可能在合并后产生风格回归；需统一跑 ruff/pytest 验证。

## 6. Open questions

1. **合并方向/载体**：路径 A（新集成分支 + 新 PR）还是路径 B（merge main 进 feature/wps-support、更新 Draft PR #1）？
2. **G2 产品决策**：合并后 qPCR **折线图**是否统一采用几何均值+log 空间误差棒（wps/1c 行为，跨宿主一致，推荐）？还是维持 main 现状（折线=算术均值，柱状=几何均值，跨宿主不一致）？
3. **PR #1 的处置**：若走路径 A，现有 Draft PR #1 是关闭（由新 PR 取代）还是保留待集成分支合并后再关闭？
4. **验证范围**：合并验证是否包含 wps-addon 的 node 测试套件（`wps-addon/tests/*.cjs`）与 Excel 真实宿主冒烟，还是仅 pytest + ruff？
5. **feat/macos-support 交互**：当前工作区检出 feat/macos-support（独立 Draft PR #2）；合并操作需切换分支。是否确认本次任务不涉及 macos 分支内容？

## 7. Persisted report

- 报告路径：`plans/explore-20260906-merge-main-wps-support.md`
- 本报告可直接作为 `/feature-plan` 的输入；所有 git 事实（merge-base、冲突文件清单、提交时间线、PR 状态）已验证，语义等价性结论（analysis.py vs presets/qpcr.py）附行号锚点。
