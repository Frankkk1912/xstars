# Plan：DC1 — qPCR log-space gate/标签双实现去重（application 委托 presets）

- 日期：2026-09-06
- 状态：**已批准并实施**（rev 1，2026-09-06 用户明确批准）
- 输入：DC1 方向决策访谈（2026-09-06，用户确认 preset 方向）+ 代码勘察锚点（本 Plan §4）
- Changelog：

| Rev | 日期 | 变更摘要 | 依据 |
| --- | --- | --- | --- |
| 1 | 2026-09-06 | 初版：基于方向访谈与全仓引用勘察生成 | interview（方向决策）+ ctx 勘察锚点 |

> 流程说明：homebrew standalone pi 下子代理基础设施受限（同前两个 Plan 的偏差记录），Plan 由编排器亲自生成。

## 1. Goal

消除 `xstars/application/analysis.py` 与 `xstars/presets/qpcr.py` 之间约 30 行的 qPCR log-space gate/标签双实现（wps 合并时按最小合并原则登记的 DC1 技术债）：**presets/qpcr.py 成为唯一实现，application/analysis.py 改为薄委托**，三宿主（Excel Windows / Excel macOS / WPS）的 qPCR 统计空间与输出标签由同一份领域代码保证，外部可观察行为完全不变（纯 REFACTOR）。

## 2. Requirements

| # | 需求 | 来源 | 硬约束 |
| --- | --- | --- | --- |
| R1 | 去重方向：presets/qpcr.py 为唯一实现；application/analysis.py 删除副本改委托 | 方向访谈 + 分层证据（§4） | 不得反向（presets → application 会制造循环依赖） |
| R2 | `application/analysis.py` 的公共 API 签名不变：`stats_input_frame(df_wide, config)`（config 版）、`qpcr_stats_table`、`PVALUE_LABEL`、`PROCESSED_DATA_SUFFIX` 对外可见性保持 | 消费方盘点（§4） | main.py ×8、application/export.py ×2、wps 链路零改动 |
| R3 | 行为等价性：重构后外部可观察行为与重构前一致，由既有双套测试锁定 | REFACTOR 验证契约 | 全量 pytest 与基线一致（0 新失败） |
| R4 | 本分支基点为 main（b10d979），独立 Draft PR，不叠加在 Draft PR #2 上 | Git 策略 | 与 PR #2 无顺序依赖（改动文件不相交） |
| R5 | 工作区处理：当前 feature/wps-support 上有未提交的 plan 格式修改，切分支前 stash，收尾恢复 | 沿用既定工作区纪律 | stash 恢复为收尾步骤 |

## 3. Non-goals

- 不修改 presets/qpcr.py 的任何实现（它是唯一实现，保持原样）。
- 不合并/删除 `stats_input_frame` 的 preset 版与 config 版两套签名（各自服务不同调用形态，均有测试锁定；签名统一属于过度设计）。
- 不触碰 main.py、export.py、plot_engine.py、wps 链路（R2 保证零改动）。
- 不处理 PR #2 / feat/macos-support 的任何待办（DC2 人工冒烟等）。
- 不修复合并中发现的与本次无关的缺陷。

## 4. Research summary（锚点均为 2026-09-06 勘察实证）

### 双实现清单（逐行等价，含常量值一致）

| 符号 | presets/qpcr.py（唯一实现） | application/analysis.py（待删副本） |
| --- | --- | --- |
| `PROCESSED_DATA_SUFFIX = " (2^-ΔΔCt)"` | L18 | L266 |
| `PVALUE_LABEL = "p-value(−ΔΔCt)"` | L19 | L267 |
| log2 变换体 | `_log2_stats_frame` L180 | 内联于 `_stats_input_frame` L234 |
| preset 版 gate | `stats_input_frame(transformed, preset)` L186 | `_stats_input_frame(transformed, preset)` L234 |
| config 版 gate | `stats_input_frame_for_config(transformed, config)` L200（enum 直查） | `stats_input_frame(df_wide, config)` L254（经 get_preset 解析）——效果等价 |
| 标签重命名 | `qpcr_stats_table` L209 | 同名副本 L270 |

application 独有（保留不动）：`_qpcr_title_suffix` L279（消费常量与 QPCRPreset，编排层职责）。

### 分层证据（R1 的决定性依据）

`application/analysis.py:21-25` 已有 5 条 `from ..presets` 导入；`presets/`、`stats_engine.py`、`plot_engine.py` 对 application 的引用为 **0**。委托方向只能是 application → presets。

### 消费方盘点（R2 的不变式清单）

| 消费方 | 用法 | 迁移后 |
| --- | --- | --- |
| `main.py` L611/1479/1507 | `_application_analysis.stats_input_frame(df, config)`（config 版） | 公共签名不变 → 零改动 |
| `main.py` L647/1483/1511 | `_application_analysis.qpcr_stats_table(...)` | re-export 保持 → 零改动 |
| `main.py` L664/1492/1518 | `_application_analysis.PROCESSED_DATA_SUFFIX` | re-export 保持 → 零改动 |
| `application/export.py` L323-325 | `from .analysis import stats_input_frame`（config 版） | 公共签名不变 → 零改动 |
| wps 链路（worker/wps_service） | 经 analyze_dataframe 等编排函数 | 内部实现变化，API 不变 → 零改动 |
| `tests/test_presets.py`、`tests/test_plot_engine.py` | 直接 import presets 版（preset 版签名、passthrough `is` 断言、常量精确值） | 零改动，天然成为等价性锁定 |

### 委托形态（实施蓝图）

```python
# application/analysis.py 头部导入区追加（别名避免与公共 config 版撞名）：
from ..presets.qpcr import (
    PVALUE_LABEL,                      # re-export（消费方经 _application_analysis.X 引用）
    PROCESSED_DATA_SUFFIX,             # re-export
    QPCROptions,
    QPCRPreset,
    qpcr_stats_table,                  # re-export
    stats_input_frame as _preset_stats_input_frame,
    stats_input_frame_for_config as _config_stats_input_frame,
)

def _stats_input_frame(transformed, preset):
    """...（docstring 指向 presets.qpcr 为唯一语义来源）"""
    return _preset_stats_input_frame(transformed, preset)

def stats_input_frame(df_wide, config):          # 公共 config 版签名不变
    """..."""
    return _config_stats_input_frame(df_wide, config)

# 删除：L266-267 常量、L270-276 qpcr_stats_table、L234-252 的 log2 实现体
# 保留：_qpcr_title_suffix（其消费的常量/QPCRPreset 改由导入提供）
```

### 行为等价性说明

两副本当前逐行等价（常量值相同、gate 判定相同：config 版经 get_preset→isinstance 与 enum 直查效果一致）。委托后唯一实现即 presets 版，test_presets.py 的 passthrough 断言（`is transformed`）与常量精确值断言直接锁定。

## 5. Gap analysis

| ID | 类型 | 内容 | 补齐任务 |
| --- | --- | --- | --- |
| G1 | REFACTOR | application/analysis.py 删除 gate/标签副本，改为委托 presets | T2.1 |
| G2 | REFACTOR | 导入区别名与 re-export 调整，公共 API 不变 | T2.1 |
| — | （横切）等价性锁定 | 既有 test_presets/test_plot_engine/application 套件即为行为等价验证，无需新测试 | T3.1 |

自查：无孤立缺口；无 FIX/SYNC/ADD 类改动（纯 REFACTOR）。

## 6. Milestone 表格

| Milestone | 内容 | 退出条件 |
| --- | --- | --- |
| M1 准备与基线 | stash wps 修改 → 从 main 新建 feat/dc1-qpcr-gate-dedup → 基线全量 pytest | 工作区 clean，基线记录 |
| M2 去重实施 | 按 §4 蓝图改造 application/analysis.py | 无冲突标记、AST 通过、grep 证实双实现仅存一份 |
| M3 验证与收敛 | 全量 pytest + compileall + git diff --check + 编辑文件 scoped ruff | 全绿（0 新失败） |
| M4 提交推送与 PR | commit + push + 新 Draft PR + 记忆更新 + 切回 wps 恢复 stash | PR 创建，工作区恢复 |

## 7. 分 milestone 的 To-do checkbox 清单

### M1 准备与基线

- [ ] T1.1 `git stash push` 保存 feature/wps-support 上的 wps plan 格式修改
- [ ] T1.2 `git switch -c feat/dc1-qpcr-gate-dedup main`（基点 b10d979）
- [ ] T1.3 基线：/tmp/xstars-venv 全量 pytest（排除 probe），记录通过/失败基线

### M2 去重实施

- [ ] T2.1 按 §4 蓝图改造 application/analysis.py：导入别名 + re-export、`_stats_input_frame`/`stats_input_frame` 改一行委托、删除 L266-267/L270-276 副本与 log2 实现体
- [ ] T2.2 核查：AST parse 通过；`grep -c "_log2_stats_frame" xstars/` 仅 presets 一处；`_qpcr_title_suffix` 与 L329/L896 的常量引用经导入解析

### M3 验证与收敛

- [ ] T3.1 全量 pytest（排除 probe）——与 T1.3 基线对比 0 新失败；test_presets qPCR stats space 套件（等价性锁定）逐条通过
- [ ] T3.2 `compileall -q xstars tests` + `git diff --check`
- [ ] T3.3 ruff check/format 限定于 `xstars/application/analysis.py`（先存债务不扩面）

### M4 提交推送与 PR

- [ ] T4.1 commit（`refactor(qpcr): deduplicate log-space gate and label helpers into presets (DC1)`）+ push
- [ ] T4.2 创建 Draft PR `feat/dc1-qpcr-gate-dedup` → `main`，描述含委托设计、等价性证据、与 PR #2 无顺序依赖说明
- [ ] T4.3 更新项目记忆（DC1 已闭合）
- [ ] T4.4 `git switch feature/wps-support` → `git stash pop`

## 8. Validation contract

| 检查项 | 方式 | 通过标准 |
| --- | --- | --- |
| 行为等价（REFACTOR） | 全量 pytest 与基线对比；test_presets qPCR stats space 套件（passthrough `is` 断言、常量精确值、log2 交叉验证）逐条通过 | 0 新失败 |
| 双实现消除 | grep 证实 `_log2_stats_frame`、log2 gate 实现体仅存在于 presets/qpcr.py | 单一实现 |
| API 不变（R2） | main.py ×8、export.py ×2 调用点零 diff；`python -c "from xstars.application.analysis import stats_input_frame, qpcr_stats_table, PVALUE_LABEL, PROCESSED_DATA_SUFFIX"` 可导入 | 导入成功、调用点无改动 |
| 分层方向（R1） | application → presets 委托；presets 无 application 导入 | 依赖方向不变 |
| 静态卫生 | compileall、git diff --check、scoped ruff | 全绿 |

## 9. 文件级修改范围 + 风险 / 回滚 / 待决事项 + Git 策略

### 文件级修改范围

| 文件 | 操作 |
| --- | --- |
| `xstars/application/analysis.py` | 唯一修改文件：导入区 + L234-281 区域委托改造 |
| `plans/20260906-dc1-qpcr-gate-dedup.md` | 本 Plan |

任何超出上表的修改需先经用户批准并修订本节（rev 升级）。

### 风险

| 等级 | 触发条件 | 影响 | 缓解 |
| --- | --- | --- | --- |
| 低 | 公共 API 意外变更（漏 re-export） | main.py/export.py NameError | R2 不变式清单 + T2.2 导入核查 + T3.1 全量测试 |
| 低 | `stats_input_frame_for_config`（enum 直查）与现行 get_preset 路径的隐性差异 | 理论等价，实测由测试锁定 | §4 等价性说明 + T3.1 |
| 低 | 循环导入（presets.qpcr ← application） | import 失败 | 方向与现行 5 条 presets 导入一致，无新增反向边 |

### 回滚

- 推送前：`git reset --hard main`（分支未合入任何内容）。
- 推送后：`git revert <commit>`；纯 REFACTOR 无数据/格式变更，回滚无副作用。

### 待决事项

- 无新增。DC1 本体即本 Plan；闭合后从项目记忆的技术债清单移除。

### Git 策略

- 分支：`feat/dc1-qpcr-gate-dedup`，基点 main（b10d979），独立 Draft PR → main。
- 与 Draft PR #2（macos）无顺序依赖：改动文件不相交（application/analysis.py、presets/qpcr.py 均为 wps 侧引入，macos 未触碰）；先合入者另一方后续 rebase/merge 即可。
- 单一 commit 承载（纯 REFACTOR，无行为变更，commit message 注明 behavior-neutral）。

## 10. 验证证据记录（实施期）

- M1 基线：分支 feat/dc1-qpcr-gate-dedup 自 main（b10d979）切出；全量 pytest（排除 probe）260 passed / 1 failed（test_standard_curve 零浓度，既录先存）。
- M2 核查：`_log2_stats_frame` / log2 gate 实现体仅存于 presets/qpcr.py（grep 实证）；diff 仅 application/analysis.py（23+/32-）；`from xstars.application.analysis import stats_input_frame, qpcr_stats_table, PVALUE_LABEL, PROCESSED_DATA_SUFFIX` 导入验证通过，QPCR config 实测 log2 gate 输出正确、常量精确值不变。
- M3 终验：全量 pytest **260 passed / 1 failed（先存）——与基线一致，0 新失败**；compileall 通过；git diff --check 干净；ruff check（0 remaining）/ format --check 均绿（uvx ruff@0.16.5，与项目收敛版本一致）。13 skipped 说明：macos 相关平台跳过在 main 基线上不存在，此基线为 main 原生套件。
- 实施期处置：PVALUE_LABEL re-export 采 redundant-alias + noqa（API 兼容，零外部消费方实证）；application/analysis.py 内 2 项先存静态建议（L400/L789）登记 defer 不扩面。
- 工作区事件：切分支前发现 pi-lens 会话间 autofix 污染 22 个文件，已随目标修改一并 stash，收尾仅提取 wps plan 文件后丢弃污染。
