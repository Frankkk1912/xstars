# Plan：合并 main 与 feat/macos-support 分支（macos 分支对齐 main 最新版）

- 日期：2026-09-06
- 状态：**已批准并实施**（rev 1 用户批准 2026-09-06；rev 2 实施期范围扩展经访谈批准）
- 输入：编排器 merge-tree 模拟诊断（2026-09-06）+ 强制访谈决策（2026-09-06，6/6 已确认）
- Changelog：

| Rev | 日期 | 变更摘要 | 依据 |
| --- | --- | --- | --- |
| 1 | 2026-09-06 | 初版：基于 merge-tree 模拟诊断与访谈决策生成 | 诊断（merge-tree ×2）+ interview（q1-q6） |
| 2 | 2026-09-06 | 实施期：实际冲突 4 文件（merge-tree 预估偏差）；CI 挂起修复（测试层 skipif）；artifacts.py np.bool_ 序列化缺陷修复；macos 测试 harness 架构适配；白名单增补 | 实施证据 + interview（CI 挂起 q1-q2、artifacts.py q1） |

> 流程偏差说明：homebrew standalone pi 下 `feature-planner` 子代理不可用（前台子代理无扩展工具 allowlist，与 plans/20260906-merge-main-wps-support.md 记录的同类偏差一致），本 Plan 由编排器亲自生成，结构契约与 wps 合并 plan 保持一致。

## 1. Goal

将 `origin/main` 最新版（b10d979，含 PR #1 WPS 支持合并 + qPCR log-space 统一 + 风格收敛）合并进 `feat/macos-support`，使 Draft PR #2 成为"main + macOS 全量"的单一合并载体：合并后 `feat/macos-support` 包含双方全部功能，macOS 自身功能（artifact 导出、rebuild payload、范围输入、CI 门禁）行为不变，qPCR 统计行为对齐 main 侧 log-space 统一版，pytest / compileall / git diff --check 全部通过，Draft PR #2 描述同步更新。

## 2. Requirements

| # | 需求 | 来源 | 硬约束 |
| --- | --- | --- | --- |
| R1 | 合并方向：在 `feat/macos-support` 上 `git merge origin/main`，更新现有 Draft PR #2，不新建 PR | 访谈 q1 | 不 rebase、不改写 macos 侧 11 个已 review commit 的历史 |
| R2 | `xstars/main.py` 采用 git 自动合并结果；若语义核查发现问题，修复必须以"macos 分支调用点 + main application 层编排共存"为语义目标，不引入第三种实现 | 诊断 + 访谈 q2 | 行级 hunk 已验证不重叠（0 冲突标记） |
| R3 | qPCR 统计行为按 main 侧为准（−ΔΔCt log2 空间假设检验、几何均值柱状/折线图、统一输出标签）——这是预期行为同步 | 访谈 q3 | macos 分支自身功能外部行为保持不变（行为等价性约束） |
| R4 | `docs/cross-platform-office-technology-strategy.md` 为 add/add（双方各自添加）：实施期 diff 两 blob（macos `1f396d7` "D8 approved copy" vs main `63882e8`），内容语义一致则保留 main 版本；存在实质差异则保留 macos D8 已批准版本并记录差异 | 访谈 q2 | 不得双份并存 |
| R5 | 验证 = 全量 pytest（排除 probe）+ compileall + `git diff --check` + macos 聚焦测试 + main.py 语义静态核查 | 访谈 q4 | 必须使用 ≥3.10 venv（/tmp/xstars-venv，Python 3.11）；`tests/test_wps_probe.py` 因 Tk 挂起必须排除（先存环境限制，非合并引入） |
| R6 | 工作区处理：切换分支前将 feature/wps-support 上未提交的 plan 格式修改 `git stash`；本任务全部完成后切回并 `stash pop` | 访谈 q5 | stash 恢复为收尾步骤，不得遗漏 |
| R7 | 本地 main（98f46ef）fast-forward 到 origin/main（b10d979） | 访谈 q2 | 仅 fast-forward，不产生新 commit |

## 3. Non-goals

- 不做 `presets/qpcr.py` 与 `application/analysis.py` 的 log2-gate/label 双实现去重（DC1 技术债，wps 合并时已登记，另开 plan 处理）。
- 不运行 `wps-addon` npm 测试（访谈 q4 未选；该目录无冲突、无手工编辑，随合并原样并入）。
- 不由 Agent 执行真实 Excel/macOS 宿主冒烟（环境不可行，仅在 PR 中标注人工验证项）。
- 不修复合并中发现的与本次合并无关的潜在缺陷（仅记录到 PR/待决事项）。
- 不修改 `ribbon/*.bas` 文件（CI 零差异门禁）。
- 不在 `feature/wps-support` 分支上做任何提交。

## 4. Research summary

### Diagnosis（阶段一诊断汇总）

**问题定性：多分支不一致（SYNC）**——`feat/macos-support`（Draft PR #2）基于旧 main（merge-base 5f4c409），落后于最新 origin/main（b10d979）约 20 个 commit（PR #1 WPS 支持合并、qPCR log-space 端口、风格收敛、docs）。此任务即 wps plan §9 预留的"PR #2 macos 随后独立处理"。

**根因与代码证据：**

- 本地 main 过时：本地 main = 98f46ef（PR #4 merge），origin/main = b10d979（PR #1 merge，晚于 PR #4）。
- `git merge-tree $(git merge-base origin/main feat/macos-support) origin/main feat/macos-support`：**0 个冲突标记**。
- 双方都改的文件（均可自动合并）：`README.md`、`README.zh-CN.md`、`xstars/main.py`。
- main 侧 main.py 改动：161 insertions / 280 deletions（重构为 application 层编排：`analyze_dataframe` / `build_analysis_writeback_plan` / `stats_input_frame`）。
- macos 侧 main.py 改动：681 insertions / 194 deletions（artifact 导出 `run_export` 区域、rebuild payload 持久化 142c832、范围输入分支）。
- 语义风险点（文本合并干净但行为可能断裂）：`_run_preset_impl`、`_run_impl`、`_run_quick_impl`、`_show_export_dialog`、`_run_standard_curve_impl` 区域双方均有 hunk。
- add/add：`docs/cross-platform-office-technology-strategy.md`（macos `1f396d7` "D8 approved copy" vs main `63882e8`）。
- 测试安全网：macos 侧 `tests/test_macos_support.py`（1223 行）+ `tests/test_artifacts.py`（469 行）；main 侧全量 260 tests。
- 先存测试基线（main@b10d979 系，macOS 本机，3.11 venv）：260 passed / 1 failed（test_standard_curve 零浓度回算 NaN，先存）；`test_wps_probe.py` Tk 挂起（先存环境限制）。

**相关 PR / commit 引用**：PR #1（MERGED，b10d979）、PR #2（DRAFT，feat/macos-support，本任务载体）、PR #4（MERGED，98f46ef）、wps 先例 merge commit 6eaa43e。

## 5. Gap analysis

| Gap | 类型 | 内容 | 正确行为依据 |
| --- | --- | --- | --- |
| G1 | SYNC | 将 origin/main 全量合入 `feat/macos-support`（WPS 支持、qPCR log-space、风格收敛） | PR #1/#4 已合并进 main，macos 分支必须对齐 |
| G2 | SYNC | qPCR 统计行为对齐 main 侧（−ΔΔCt 空间、几何均值、统一标签） | 访谈 q3 已确认接受 |
| G3 | SYNC | `xstars/main.py` 合并后语义核查：macos 调用点（artifact/rebuild/range）与 main application 层编排共存 | R2；测试锁定 |
| G4 | SYNC | add/add 策略文档收敛为单一版本 | R4 |
| G5 | REFACTOR | 本地 main fast-forward（仓库卫生，无代码影响） | R7 |

## 6. Milestone 表格

| Milestone | 内容 | 退出条件 |
| --- | --- | --- |
| M1 准备与基线 | stash wps 修改 → 切到 feat/macos-support → 记录 pre-merge SHA → ff 本地 main → macos 聚焦测试基线 | 工作区 clean，基线记录 |
| M2 合并与核查 | merge origin/main → 核对自动合并结果与 merge-tree 预期 → add/add 文档收敛 → main.py 语义静态核查（必要时修复） | 合并 commit 语义正确 |
| M3 验证与收敛 | 全量 pytest（排除 probe）+ compileall + git diff --check + macos 聚焦复跑 | 全绿（允许先存失败） |
| M4 提交推送与 PR 更新 | push → 更新 Draft PR #2 body（含 Sync with main 小节）→ 切回 wps 分支 stash pop | PR 更新，工作区恢复 |

## 7. 分 milestone 的 To-do checkbox 清单

### M1 准备与基线

- [ ] T1.1 `git stash push` 保存 feature/wps-support 上的 plan 格式修改（记录 stash ref）
- [ ] T1.2 `git switch feat/macos-support`，确认工作区 clean（plan 文件为 untracked 随行）
- [ ] T1.3 记录 pre-merge SHA：`git rev-parse HEAD`（预期 50ead19）
- [ ] T1.4 `git branch -f main origin/main` 之外的等价 fast-forward（`git fetch` 后 `git branch -f main origin/main` 或在 main 上 ff），使本地 main = b10d979
- [ ] T1.5 基线：/tmp/xstars-venv 跑 `pytest tests/test_macos_support.py tests/test_artifacts.py -q`，记录结果

### M2 合并与核查

- [ ] T2.1 `git merge origin/main`（预期 0 冲突自动合并；若出现意外冲突，停下记录并按 R2/R4 语义解决）
- [ ] T2.2 核对合并树与 merge-tree 预期一致（changed-in-both 三文件 + add/add 策略文档，无意外文件）
- [ ] T2.3 add/add 处理：`git diff 63882e8 1f396d7 -- docs/cross-platform-office-technology-strategy.md`（blob 对比），按 R4 规则收敛并记录决策
- [ ] T2.4 main.py 语义静态核查：合并后对 `_run_preset_impl` / `_run_impl` / `_run_quick_impl` / `_show_export_dialog` / `_run_standard_curve_impl` 区域核查 macos 分支调用点（artifact 导出、rebuild payload、range 输入）未被 application 层重构截断；核对 `import` 区与模块头合并正确
- [ ] T2.5 若 T2.4 发现语义断裂：最小修复（限定 main.py 白名单），commit message 显式描述旧行为/新行为/原因
- [ ] T2.6 生成合并 diff 快照（name-status + 统计）供核查记录

### M3 验证与收敛

- [ ] T3.1 `/tmp/xstars-venv/bin/python -m pytest tests -q --ignore=tests/test_wps_probe.py`（基线 260 passed / 1 failed 先存；合并后 macos 侧测试并入，总数上升）
- [ ] T3.2 `/tmp/xstars-venv/bin/python -m compileall -q xstars tests`
- [ ] T3.3 `git diff --check`（合并 commit 范围）
- [ ] T3.4 macos 聚焦复跑：`pytest tests/test_macos_support.py tests/test_artifacts.py -q` 与 T1.5 基线对比（行为等价性证据）
- [ ] T3.5 若 T2.5 有手工编辑：对被编辑文件跑 ruff check（先存债务不扩面）

### M4 提交推送与 PR 更新

- [ ] T4.1 `git push origin feat/macos-support`（merge commit + 必要修复 commit）
- [ ] T4.2 提交本 Plan（`docs(plan): add merge-main-macos-support plan`）并 push
- [ ] T4.3 更新 Draft PR #2 body：新增 "Sync with main (2026-09-06)" 小节（合并内容、add/add 决策、语义核查结论、验证证据、人工冒烟标注项）
- [ ] T4.4 `gh pr view 2` 确认 Draft 状态保持、head 正确
- [ ] T4.5 收尾：`git switch feature/wps-support` → `git stash pop` 恢复 T1.1 修改（若恢复冲突则停下报告）

## 8. Validation contract

| Gap 类型 | 验证方式 |
| --- | --- |
| G1 SYNC（分支对齐） | `git merge-base --is-ancestor origin/main feat/macos-support` 为真；merge 树文件清单与 merge-tree 预期一致（T2.2/T2.6） |
| G2 SYNC（qPCR 行为） | main 侧 qPCR 测试套件全绿（T3.1 中 test_qpcr*/test_presets/test_application_analysis 相关用例）；标签/几何均值行为以 main@b10d979 测试为准 |
| G3 SYNC（main.py 语义） | T2.4 静态核查 + T3.4 macos 聚焦测试复跑与基线一致（行为等价性证据） |
| G4 SYNC（add/add） | T2.3 blob diff 结论记录，仓库中该文件仅一个版本 |
| G5 REFACTOR（main ff） | `git rev-parse main` = b10d979 |
| 整体行为等价 | T3.1 全量 pytest 与先存基线一致（无新增失败）；T3.4 macos 测试与 T1.5 基线一致 |

**允许的先存失败**：test_standard_curve 零浓度回算 1 failed；test_wps_probe.py 排除不跑。两者必须在 macos 分支 pre-merge 基线上同样存在（若不是，即为合并引入，必须修复）。

## 9. 文件级修改范围 + 风险 / 回滚 / 待决事项 + Git 策略

### 文件级修改范围

| 文件 | 操作 | 来源 |
| --- | --- | --- |
| 全部合并引入文件 | merge 自动合并 + 4 文件冲突解决 | G1 |
| `xstars/main.py` | 冲突解决（application 层流程 + macos graft + 类型对齐 + graft 设计修正） | G3 |
| `docs/cross-platform-office-technology-strategy.md` | add/add 收敛为 main 版 | G4 |
| `plans/20260906-merge-main-macos-support.md` | 新增本 Plan | T4.2 |
| Draft PR #2 body | 文档更新 | T4.3 |
| `tests/test_wps_probe.py`、`tests/test_excel_characterization.py` | **rev 2 增补**（访谈批准）：darwin 平台 skipif（COM/Tk 专属契约在 darwin 不可达，否则 macos CI 挂起） | 实施期访谈 q1 |
| `tests/test_macos_support.py` | **rev 2 增补**（访谈批准）：harness mock 目标适配 application 层架构（StatsEngine/transform_dataframe patch 目标、WritebackPlan 写表断言） | 同上 |
| `xstars/artifacts.py` | **rev 2 增补**（访谈批准）：_json_safe 增加 np.bool_ 分支；_stats_to_dict 两字段包 _json_safe——修复先存生产缺陷（真实统计时重建 payload 静默注册失败） | 实施期访谈 q1 |

任何超出上表的项目代码修改需先经用户批准并修订本节（rev 升级）。

### 风险

- **语义冲突残留**（中）：文本 0 冲突不代表语义正确；main.py 双侧大改（161+/280- vs 681+/194-），靠 T2.4 静态核查 + T3.4 行为等价测试兜底。
- **二进制文件**（低）：`XSTARS_Templates.xlsx` 仅 main 侧改动（不在 changed-in-both 清单），自动取 main 版；若 T2.2 发现其出现在双方改动清单则停下来单独核对模板内容。
- **测试环境**（低）：Python 3.9 不可用、probe Tk 挂起——均为已记录先存限制，使用 3.11 venv 并排除 probe。
- **stash 恢复冲突**（极低）：stash 内容为 plan 文档格式微调，与 macos 分支无交集。

### 回滚

- 合并未推送前：`git reset --hard <pre-merge SHA>`（T1.3 记录值）。
- 已推送后：`git revert -m 1 <merge-commit>`，PR #2 保持 Draft 不受影响。
- 本地 main ff 回滚：`git branch -f main 98f46ef`。

### 待决事项

- DC1：presets/qpcr.py 与 application/analysis.py 的 log2-gate/label 双实现去重（延后，另开 plan）。
- DC2：真实 macOS 宿主（Excel for Mac）人工冒烟——PR 中标注为人工验证项。
- DC3：add/add 策略文档若存在实质差异，具体取舍结论在 T2.3 记录后回填本节。

### Git 策略

- 单一 Draft PR #2 承载全部 Milestone，不拆分。
- 合并方向：`git merge origin/main`（非 rebase / 非 cherry-pick），与 wps 先例 6eaa43e 同构。
- commit 约定：merge commit 默认 message 或 `Merge remote-tracking branch 'origin/main' into feat/macos-support`；手工修复（如有）单独 commit 并显式描述行为变更；Plan 提交用 `docs(plan):` 前缀。
- `ribbon/*.bas` 零差异门禁持续有效。

## 10. 验证证据记录（实施期）

- M1 基线：pre-merge SHA `50ead19`；macos 聚焦测试（test_macos_support + test_artifacts）88 passed；本地 main fast-forward 至 b10d979。
- M2 核查：实际合并 4 文件冲突（README.md、README.zh-CN.md、策略文档 add/add、xstars/main.py 20 区）——merge-tree 预估为 0 冲突，与实际不符（教训：merge-tree 不能替代实际 merge 验证）。解决语义见 merge commit 690a733 正文。add/add blob diff 确认仅编辑注差异，取 main 版（R4）。语义 graft：_register_writeback_artifacts 初版按计划名匹配迭代失败（Excel 重命名图片为 {name}_Final），修正为 _execute_analysis_writeback 返回插入图片对象后按序注册。
- M3 终验（/tmp/xstars-venv，Python 3.11.15）：全量 pytest（含 skipif 后）**353 passed / 13 skipped / 0 failed**；compileall 通过；git diff --check 干净；macos 聚焦测试全绿。13 skipped = darwin 平台 COM/Tk 专属契约（probe elisa/shape/com ×9、characterization ×2、其余为既有平台跳过）。
- 实施期缺陷修复：artifacts.py np.bool_ 序列化（先存生产缺陷，真实统计时重建 payload 静默注册失败；合并架构使测试显形）。
- 依赖说明：验证使用 /tmp/xstars-venv；未改动仓库外环境。
