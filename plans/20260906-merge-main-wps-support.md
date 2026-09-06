# Plan：合并 main 与 feature/wps-support 分支

- 日期：2026-09-06
- 状态：**已批准**（rev 3，2026-09-06 用户明确批准）
- 输入：`plans/explore-20260906-merge-main-wps-support.md`（explore 报告）+ 强制访谈决策（2026-09-06，4/4 已确认）
- Changelog：

| Rev | 日期 | 变更摘要 | 依据 |
| --- | --- | --- | --- |
| 1 | 2026-09-06 | 初版：基于 explore 报告与访谈决策生成 | explore 报告 + interview（q1-q4） |
| 2 | 2026-09-06 | 实施期 review 修复 F1；验证偏差记录 | fresh-context review round 1 |
| 3 | 2026-09-06 | Round 2 review 收尾：§6/§8 验证措辞对齐实际证据、锚点校正、待决事项 DC3/DC4 登记、T2.3 cast 表述澄清 | fresh-context review round 2 |

## 1. Goal

将 `main`（含已合并的 PR #4 qPCR log-space 工作）合并进 `feature/wps-support`，解决 4 个冲突文件并统一 qPCR 折线图行为，使 Draft PR #1 成为"main + WPS 全量"的单一合并载体：合并后 `feature/wps-support` 包含双方全部功能，pytest / wps-addon node 测试 / ruff 全部通过，Draft PR #1 描述同步更新。

## 2. Requirements

| # | 需求 | 来源 | 硬约束 |
| --- | --- | --- | --- |
| R1 | 合并方向为路径 B：在 `feature/wps-support` 上 `git merge main`，更新现有 Draft PR #1，不新建 PR | 访谈 q1 | 不 rebase、不改写 wps 历史 |
| R2 | 冲突解决语义：`xstars/main.py` 取 wps application 层调用（11 处冲突块）；`tests/test_presets.py` 取 main 超集；`XSTARS_Templates.xlsx` 取 wps 对齐版 | explore §4/G1/G3/G4 | 不得引入第三种实现 |
| R3 | qPCR 折线图统一为几何均值 + log 空间误差棒（保留 wps `_line` gate，叠 main 的 `cast(Any)` 风格） | 访谈 q2 | 行为以 1c19a63 为准 |
| R4 | 验证 = pytest 全量 + `wps-addon` node 测试 + ruff check/format | 访谈 q3 | 真实宿主冒烟仅作人工标注项 |
| R5 | 本次不涉及 `feat/macos-support` / PR #2 内容；仅切换分支执行合并 | 访谈 q4 | 工作区从 feat/macos-support 切出前必须 clean |
| R6 | 双 gate/logic 的重复实现（`presets/qpcr.py` vs `application/analysis.py`）在本合并中原样保留，不重构 | explore §4（最小合并原则） | 重构去重仅记录为技术债/待决事项 |

## 3. Non-goals

- 不修改 `feat/macos-support` 分支及其 Draft PR #2。
- 不 rebase / 不改写 `feature/wps-support` 既有 53 个提交。
- 不由 Agent 执行真实 Excel/WPS 宿主冒烟（环境不可行，仅在 PR 中标注人工验证项）。
- 不做 presets/qpcr.py 与 application/analysis.py 的去重重构（R6）。
- 不修复合并中发现的与本次合并无关的潜在缺陷（仅记录到 PR/待决事项）。
- 不改 wps-addon 前端 JS 代码（该目录无冲突，随合并原样并入）。

## 4. Research summary

**代码库现状**（均已在 explore 阶段验证，锚点见报告）：

- 分支拓扑：merge-base `5f4c409`；main +12（PR #4）、wps +53（+20254 行）；`git merge-tree` 实测恰好 4 个冲突文件：`XSTARS_Templates.xlsx`、`tests/test_presets.py`、`xstars/main.py`（11 处冲突块）、`xstars/plot_engine.py`；双方同时修改的文件仅此 4 个。
- 语义等价性：wps 侧 `application/analysis.py:234-280` 已包含与 main `presets/qpcr.py:168-199` 语义一致的 log2 gate、`PVALUE_LABEL`、`qpcr_stats_table`（由 `274e169` 对齐 PR #4）→ main.py 冲突可整体取 wps 结构。
- 模板二进制：wps 版（274e169，09-03 20:23）晚于 main 版（eaa9f10，09-03 18:17）且声明对齐；unzip 对比两侧 sheet3 均含 27.7/28.4 修复值 → wps 版为超集。
- 折线图分歧：wps `plot_engine.py:225-233` 的 `_line` qPCR 几何均值 gate 来自 1c19a63 原始实现；main PR #4 未移植，且其 Plan Non-goal 10（`plans/20260903-qpcr-log-space-excel-sync.md:53`）"1c 只特化 BAR_SCATTER"与 1c19a63 实际代码（`git show 1c19a63:xstars/plot_engine.py:225`）矛盾。访谈决策：取 wps 行为。
- 测试：main `tests/test_presets.py` 为超集（+308 行，含 log-space 交叉验证 :590-647），wps 侧 delta（+17/-5：import 重排 + `analyze(log2_df)` 断言）已被覆盖；`tests/test_plot_engine.py` 仅 main 修改（+77），无 qPCR 折线图断言，与 R3 不冲突；两侧均无 `_line` qPCR 回归测试（缺口 → T3.1）。
- 合并后重复实现：`presets/qpcr.py`（main 版，被 main 测试引用）与 `application/analysis.py`（wps 版，被应用层引用）将并存约 30 行等价逻辑（R6 保留）。
- PR/CI：PR #1（`feature/wps-support → main`）为 open Draft；双侧均无 GitHub Actions workflow；wps-addon 测试命令为 `npm test`（= `node --test tests/*.test.cjs`，wps-addon/package.json scripts.test）。

**外部方案**：纯 git 合并操作，无外部 API/库引入；git 官方 merge/merge-tree 语义为既有实践，无需外部调研。证据缺口：无。

**推荐方案**：路径 B（R1）+ §R2/R3 冲突语义 + T3.1 补回归测试 + 三套自动化验证。

## 5. Gap analysis

| ID | 功能缺口 | 现状 | 影响 | 补齐任务 |
| --- | --- | --- | --- | --- |
| G1 | main.py 结构冲突（内联编排 vs application 层） | 11 处冲突块（merge-tree 实测） | 无法自动合并 | T2.2 |
| G2 | qPCR 折线图跨宿主行为分歧 | main=算术均值，wps/1c=几何均值（plot_engine.py:225-233 vs main 无 gate） | 同一数据两侧图形不一致 | T2.3, T3.1 |
| G3 | test_presets.py 双侧分叉 | main 超集 vs wps 小 delta（已被覆盖） | 合并归属需裁决 | T2.4, T3.2 |
| G4 | 模板二进制无法三方合并 | 两版均存在；wps 版为对齐超集 | 择一错误则模板示例退化 | T2.1 |
| G5 | 合并后风格/静态检查回归风险 | 两侧各自做过 ruff 收敛，合并体未经统一检查 | CI（未来）或 review 阻塞 | T3.4 |
| — | （横切）合并前基线未记录 | wps tip 的测试状态无快照 | 无法区分"先存失败"与"合并引入" | T1.2（支撑 R4 验证契约） |
| — | （横切）PR #1 描述未反映合并内容 | Draft PR 描述停留在合并前状态 | review 者缺少冲突解决说明 | T4.2（支撑 R1 交付） |

**自查：无孤立缺口** —— G1→T2.2、G2→T2.3+T3.1、G3→T2.4+T3.2、G4→T2.1、G5→T3.4；横切项 T1.2/T4.2 分别注明支撑 R4 与 R1。

## 6. Milestone 表格

| Milestone | Status | Dependencies | Validation | Notes |
| --- | --- | --- | --- | --- |
| M1 准备与基线 | [x] | 无 | 工作区 clean、分支正确（`git status` / `git log -1`）；基线测试结果已记录 | 从 feat/macos-support 切出（R5） |
| M2 合并与冲突解决 | [x] | M1 | 4 文件按 R2/R3 语义解决；`git diff --check` 无冲突标记；stale import 核查通过 | 唯一写入者，串行执行 |
| M3 验证与收敛 | [x] | M2 | pytest 相对基线无新失败；node 测试绿；合并涉及文件 scoped ruff 绿 | 真实宿主冒烟不在本机（R4） |
| M4 提交推送与 PR 更新 | [x] | M3 | 合并提交落盘并推送；Draft PR #1 描述已更新含合并说明 | PR 描述草稿见 §9 |

## 7. 分 milestone 的 To-do checkbox 清单

### M1 准备与基线

- [x] T1.1 同步远端并切换分支
  - 文件：无（git 操作）
  - 修改：`git fetch origin`；确认工作区 clean 后 `git switch feature/wps-support`（本地不存在则 `git switch -c feature/wps-support origin/feature/wps-support`）；`git pull --ff-only`
  - 验收：`git log --oneline -1` == `93b1741`（wps tip）；`git status` clean
  - 依赖：无
- [x] T1.2 记录合并前基线
  - 文件：无（仅记录到本 Plan 完成汇报）
  - 修改：在 wps tip 运行 `python -m pytest tests -q` 与 `wps-addon` 下 `npm test`，记录通过/失败基线
  - 验收：基线结果已记录；若存在先存失败，逐条标注"先于合并存在"
  - 依赖：T1.1

### M2 合并与冲突解决

- [x] T2.1 解决 `XSTARS_Templates.xlsx`（G4）
  - 文件：`XSTARS_Templates.xlsx`
  - 修改：`git checkout --ours XSTARS_Templates.xlsx`（合并方向为 main→wps，ours=wps 对齐版 274e169）+ `git add`
  - 验收：unzip 后 sheet3 含 27.7/28.4 Ct 值；文件与 wps tip 版本逐字节一致（`git diff origin/feature/wps-support -- XSTARS_Templates.xlsx` 为空）
  - 依赖：T2.0（见下）
- [x] T2.0 启动合并
  - 文件：无
  - 修改：`git merge main --no-ff --no-commit`（预期停在 4 文件冲突）
  - 验收：`git status` 显示恰好 4 个 unmerged 文件
  - 依赖：T1.2
- [x] T2.2 解决 `xstars/main.py`（G1）
  - 文件：`xstars/main.py`
  - 修改：11 处冲突块全部取 theirs 语义后手工校验——#4/#6/#7 取 wps `analyze_dataframe`/`build_analysis_writeback_plan` 流程；#5/#10/#11 取 `_application_analysis.stats_input_frame(...)`；#1/#2/#3/#8/#9 取 wps 宿主无关结构（动态 `import_module("ttkbootstrap")`、`Any` 注解、import 合并）；随后核对 main 侧独有 import（`stats_input_frame`/`stats_input_frame_for_config` from presets 等）在保留代码中是否仍被引用，删除未引用项
  - 验收：文件无冲突标记；`python -c "import ast; ast.parse(open('xstars/main.py').read())"` 通过；grep 无 `<<<<<<<`/`>>>>>>>`
  - 依赖：T2.0
- [x] T2.3 解决 `xstars/plot_engine.py`（G2，R3）（澄清：cast(Any) 对齐 main tip 实际位置 = _qpcr_bars 内调用行；_line gate 内不加 cast）
  - 文件：`xstars/plot_engine.py`
  - 修改：保留 wps 侧 `_line` qPCR gate（:225-233 区域），将 `_qpcr_geo_stats` 调用行对齐 main 的 `cast(Any, ...)` 写法；import 行取 `from typing import TYPE_CHECKING, Any, cast`
  - 验收：与两侧 tip 的差异仅剩 cast 风格；`_line` gate 存在；无冲突标记
  - 依赖：T2.0
- [x] T2.4 解决 `tests/test_presets.py`（G3）
  - 文件：`tests/test_presets.py`
  - 修改：取 main 版本（`git checkout --theirs tests/test_presets.py` + `git add`）
  - 验收：文件与 main tip 版本一致（`git diff main -- tests/test_presets.py` 为空）
  - 依赖：T2.0
- [x] T2.6 引用一致性核查
  - 文件：`xstars/main.py`（如核查发现残留则修正）
  - 修改：grep 确认保留代码引用的所有符号均有 import；确认 `presets/qpcr.py` 的 gate 函数仅被 main 版测试引用（R6 并存）
  - 验收：无 `NameError` 级引用缺失（由 T3.2 pytest 兜底确认）
  - 依赖：T2.1, T2.2, T2.3, T2.4

### M3 验证与收敛

- [x] T3.1 新增 qPCR 折线图回归测试（G2 锁定）
  - 文件：`tests/test_plot_engine.py`
  - 修改：新增 `test_qpcr_line_uses_geometric_means`：构造 qPCR preset 配置与已知数据，断言 `_line` 产生的 means 为几何均值（对比 `np.exp(np.mean(np.log(...)))`）且误差棒不对称
  - 验收：新测试通过；`git diff main -- tests/test_plot_engine.py` 仅含该新增测试
  - 依赖：T2.3
- [x] T3.2 pytest 全量（G3 回归）（验证环境 Python 3.11 venv；1 个 standard_curve 失败与 wps probe 挂起均在 wps tip 逐字节复现 = 先存，非合并引入）
  - 文件：无
  - 修改：`python -m pytest tests -q`
  - 验收：0 failed（与 T1.2 基线对比，不得出现新失败）
  - 依赖：T2.6
- [x] T3.3 wps-addon node 测试（R4）
  - 文件：无
  - 修改：`cd wps-addon && npm test`
  - 验收：node --test 全部通过（与 T1.2 基线一致）
  - 依赖：T2.0
- [x] T3.4 ruff 风格收敛（G5）（合并编辑文件零报错；全仓 109 项为 tip 先存债务，超出批准范围不修）
  - 文件：仓库 Python 文件（如 ruff 报告需要修正）
  - 修改：`ruff check .` 与 `ruff format --check .`；如有可自动修复项，仅修复合并引入的偏差
  - 验收：两命令均零报错
  - 依赖：T2.6

### M4 提交推送与 PR 更新

- [x] T4.1 提交合并并推送
  - 文件：无（git 操作）
  - 修改：`git commit`（合并信息：`Merge main into feature/wps-support: reconcile qPCR log-space port (PR #4)`，正文列 4 文件解决语义与 `_line` 统一决策）；`git push origin feature/wps-support`
  - 验收：`git log -1` 为合并提交且双亲正确；push 成功
  - 依赖：T3.1, T3.2, T3.3, T3.4
- [x] T4.2 更新 Draft PR #1 描述（R1 交付）（PR #1 已追加 Sync with main 小节，PR 保持 OPEN+Draft）
  - 文件：无（`gh pr edit 1`）
  - 修改：在 PR 描述追加"与 main 同步（2026-09-06）"小节：4 文件解决语义、`_line` 统一决策及依据、双实现并存技术债说明、真实宿主冒烟为人工验证项
  - 验收：`gh pr view 1` 含新增小节；PR 仍为 Draft
  - 依赖：T4.1

## 8. Validation contract

| 检查项 | 命令/方式 | 预期结果 | 通过标准 |
| --- | --- | --- | --- |
| 冲突清除 | `git diff --check`；`grep -rn '<<<<<<<\|>>>>>>>' xstars/ tests/` | 无输出 | 零冲突标记 |
| 模板正确性 | `git diff origin/feature/wps-support -- XSTARS_Templates.xlsx` | 空 | 与 wps 对齐版逐字节一致 |
| main.py 语法 | `python -c "import ast; ast.parse(open('xstars/main.py').read())"` | 无异常 | 可解析 |
| 折线统一（R3） | `pytest tests/test_plot_engine.py -q`（含新增 T3.1） | 全绿且新测试断言几何均值 | `_line` gate 存在且回归测试通过 |
| Python 回归 | `python -m pytest tests -q` | 0 failed 或仅先存失败（与 T1.2 基线对比无新失败） | 0 failed 或仅先存失败（与 T1.2 基线对比无新失败） |
| 前端回归 | `cd wps-addon && npm test` | 全绿 | node --test 0 failed |
| 静态风格 | `ruff check .` + `ruff format --check .`（合并编辑文件必须零报错；全仓先存债务不要求清零） | 合并编辑文件零报错 | scoped 双命令通过；全仓先存债务已记录 |
| PR 状态 | `gh pr view 1 --json state,title,body` | OPEN + DRAFT + 含合并小节 | 描述更新且未误关闭/误 ready |
| 人工验证（标注项） | 真实 Excel/WPS 宿主 qPCR 柱状+折线冒烟 | 由具备 Windows/WPS 环境者执行 | 本环境不可行，PR 中标注为人工验证项，责任人在 PR 描述中指派 |

## 9. 文件级修改范围 + 风险 / 回滚 / 待决事项 + Git 策略

### 文件级修改范围

| 文件 | 操作 | 说明 |
| --- | --- | --- |
| `XSTARS_Templates.xlsx` | 冲突解决（取 wps 版） | G4 |
| `xstars/main.py` | 冲突解决（wps 结构 + import 核对） | G1 |
| `xstars/plot_engine.py` | 冲突解决（wps `_line` gate + main cast 风格） | G2/R3 |
| `tests/test_presets.py` | 冲突解决（取 main 版） | G3 |
| `tests/test_plot_engine.py` | 修改（新增 1 个回归测试） | G2 锁定 |
| 其余约 +20k 行（`wps-addon/`、`xstars/application/`、`xstars/wps_service.py`、`ribbon/` 等） | 自动并入，不手工改动 | wps 独有新增 |
| `xstars/presets/qpcr.py` | 随 main 版自动并入，不手工改动 | R6 并存保留 |
| `feat/macos-support` 相关文件 | 明确不修改 | R5 |

### 风险

| 等级 | 触发条件 | 影响 | 缓解 |
| --- | --- | --- | --- |
| 中 | main.py 11 处冲突块手工校验遗漏 main 侧行为（如某调用点参数差异） | 合并后 Excel 宿主路径行为回退 | R2 逐块语义清单 + T2.6 引用核查 + T3.2 全量测试 |
| 中 | 双 gate 实现并存（R6）后续单侧演化 | 两侧统计行为再次漂移 | PR 描述标注技术债；待决事项 DC1 |
| 低 | node/pytest 环境在 macOS 本机与 CI（未来）差异 | 验证结果不可复现 | T1.2 记录基线环境；PR 标注验证环境 |
| 低 | `cast(Any)` 风格混入引发 ruff format 分歧 | T3.4 报错 | ruff 双命令兜底，仅修合并引入偏差 |

### 回滚

- 合并未推送前：`git merge --abort`（T2.0 阶段）或 `git reset --hard 93b1741`（M2 完成后、推送前）。
- 已推送后：`git revert -m 1 <merge-commit>` 生成反向提交推回（保留历史，不 force-push）；数据/配置/API 兼容性：合并提交本身不引入持久化数据变更，回滚仅影响代码状态；Draft PR #1 描述需相应追加回滚说明。
- 不使用 force-push（PR #1 已有 review 记录，须保留）。

### 待决事项

- DC1（技术债）：`presets/qpcr.py` 与 `application/analysis.py` 的 gate/标签逻辑并存。本 Plan 按 R6 保留；去重方向（application 层委托 presets 或反向）留待后续独立 Plan。
- DC2（人工验证责任人）：真实 Excel/WPS 宿主冒烟由谁执行、何时执行——PR 中标注，待用户指派。
- DC3（仓库卫生，延期）：`.gitignore` 的 `*.xlsx` 规则遮蔽 `XSTARS_Templates.xlsx`（`.gitignore:22`），且无 `.gitattributes` binary 声明；建议后续独立 PR 增 `!XSTARS_Templates.xlsx` 反白与 binary 属性。
- DC4（测试缺口，延期）：`main.py` 三条内联 qPCR 写回路径的标签 wiring 无自动化回归测试（Round 1 F1 即由此静默丢失）；候选方案为 `test_excel_characterization.py` 增加 mock-book 特征测试或源码级守卫测试；需用户批准后另行实施。

### Git 策略

- **分支名**：复用既有 `feature/wps-support`（R1，不新建 feat/ 分支；slug 对应 `merge-main-wps-support`，Plan/探索报告落在 `plans/`）。
- **Draft PR**：更新 PR #1，标题不变（`feat: add Windows WPS support`）。
- **PR 描述草稿（追加小节）**：

  > ## Sync with main (2026-09-06)
>
  > Merged `main` (incl. PR #4 qPCR log-space port) into this branch. Conflict resolutions:
  > - `xstars/main.py` — adopted the application-layer flow (`analyze_dataframe` / `build_analysis_writeback_plan` / `stats_input_frame`); the inline PR #4 call sites are covered by the aligned equivalents added in 274e169.
  > - `xstars/plot_engine.py` — kept the 1c19a63 `_line` qPCR geometric-mean gate; unified line-chart behavior across Excel and WPS hosts (decision: qPCR line charts show geometric means with log-space error bars).
  > - `tests/test_presets.py` — took the main (superset) version; wps delta already covered.
  > - `XSTARS_Templates.xlsx` — took the 274e169 aligned template (superset of the PR #4 fix).
  > - Added `test_qpcr_line_uses_geometric_means` regression test.
  > Known tech debt: log2-gate/label helpers now exist in both `xstars/presets/qpcr.py` and `xstars/application/analysis.py` (kept verbatim for minimal merge; dedup deferred to a follow-up plan). Real-host (Excel/WPS) smoke test is a manual acceptance item.
- **PR 拆分决策**：单一 PR（PR #1）承载全部合并与冲突解决——本任务为单一分支合并，Milestone 全部串行进入同一 PR，不满足拆分升级条件（Milestone <5 或非跨独立子系统新增）。
- **合并顺序**：PR #1（本合并完成后）合入 `main`；PR #2（macos）随后独立处理，与本任务无顺序依赖。

## 10. 验证证据记录（实施期）

- M1 基线：系统 Python 3.9 下 `python -m pytest tests -q` 出现 2 个 `xlwings` collection error；`cd wps-addon && npm test` 为 29 passed。
- M3 终验（Python 3.11.15 venv）：pytest 为 260 passed / 1 failed（先存）；`test_wps_probe.py` 为 5 过 / 1 挂起（macOS Tk 环境限制，在 wps tip 复现）；`cd wps-addon && npm test` 为 29 passed，退出码 0；ruff scoped 检查退出码 0；`git diff --check` 干净。
- 依赖说明：验证使用 `/tmp/xstars-venv`，通过 `uv` 安装 requirements、pytest 与 pytest-timeout；未改动仓库。
- Round 1 review：F1 已在 `c28501f` 修复，恢复 Excel 三条内联 qPCR 写回路径的 `p-value(−ΔΔCt)` 与 ` (2^-ΔΔCt)` 标签；聚焦测试 103 passed，AST 与 6 处标签引用核查通过；全文件 ruff 仍因 wps tip 先存的 24 项 lint 与格式债务失败，为避免扩大范围未自动格式化。F2 已澄清为无行为影响的 cast 放置，F4 因现有端到端/Excel characterization 测试覆盖而降级；F3 双实现与 F5 xlsx 仓库卫生分别记录为技术债和范围外延期。
- Round 2（3 reviewer 并行：coverage/correctness/maintainability）：零 Blocker。已核无问题：T3.1 测试有效锁定几何均值行为（真实 PlotEngine 路径无 mock）、test_presets main 版对 wps delta 强于原断言覆盖、F1 四 hunk 位置/gate/作用域正确且与 main tip 语义逐条一致、双实现 R6 原样保留属实、无夹带改动（合并树 vs 纯自动合并树差异恰为 3 个文本冲突文件；模板 blob 5dfdd020 与 wps tip 一致）。终版证据：F1 后全量 pytest（3.11 venv，排除 probe）260 passed/1 failed（先存）。F-1 守卫测试缺口登记为 DC4（避免超批准范围）；cast 位置按 main tip 保留并在 T2.3 澄清；F-3 docstring 注释、F-6 重复测试去重、F-5 别名统一均登记/忽略（理由见 review 记录）。
