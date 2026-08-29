# XSTARS macOS 开发者模式 MVP 实施 Plan

- **状态**：已批准（2026-08-29 用户会话确认，rev 2026-08-29）
- **计划日期**：2026-08-29
- **目标分支**：`feat/macos-support`
- **目标 PR**：单个 Draft PR
- **实施状态说明**：本文仅为实施计划；所有 Milestone 与 To-do 初始均未完成。

## 1. Goal

在不改变 Windows 现有行为、且不修改任何现有 `ribbon/*.bas` 文件的前提下，使 XSTARS 能以开发者模式运行于 macOS 10.14+、Excel for Mac 2016+、Intel 与 Apple Silicon 环境：用户可复用现有 RunPython VBA 回调生成图表；可为 XSTARS 生成并已登记重建信息的图表导出高清图片；标准曲线/ELISA 需要第二选区时可通过 tkinter 手动输入范围；安装与实机验收步骤可复现。

完成结果必须同时满足：

1. macOS 平台分支的自动化单元测试通过，完整测试套件无回归，静态语法检查通过。
2. Windows 的 COM Shape 导出、ShapeRange 选择、Ribbon/VBA 文件及现有用户行为保持不变。
3. macOS Export 不再依赖 Excel COM、`CopyPicture` 或 `PIL.ImageGrab`，而是加载生成时持久化的 payload、重建 Matplotlib Figure，再以 `fig.savefig()`/现有 `export_figure()` 输出。
4. macOS Excel 实机清单由用户执行并把结果记录到 Draft PR；未取得实机记录前不得把 PR 标记为 Ready for Review。

## 2. Requirements

### 硬约束

1. **R1 — MVP 范围**：仅交付 Python 侧跨平台修复、现有 RunPython 模式 VBA 复用、macOS 安装文档。独立 `.app` 分发留待后续 PR。  
   **来源**：2026-08-29 用户访谈决策 1。
2. **R2 — Export 范围**：macOS 只支持 XSTARS 生成且存在有效重建 payload 的图表；通过 Python 重建 Figure 后调用 `fig.savefig()`，不得从 Excel 反向截图；不支持任意 Excel Shape。  
   **来源**：2026-08-29 用户访谈决策 2；后续批准的 Export 方案 A。
3. **R3 — Export payload**：生成图表时同步持久化可重建 payload（处理后数据、专用 `PrismConfig` 快照、图种/渲染器信息、统计结果或专用曲线参数），以图片名关联保存到本地目录（建议 `~/.xstars/artifacts/`）；Export 时加载 payload 并重建 Figure。  
   **来源**：2026-08-29 用户对 Export 方案 A 的明确批准。
4. **R4 — best-effort 写入**：payload 写入失败不得阻断统计、出图或插入 Excel；失败必须可诊断。macOS Export 遇到缺失、损坏或不兼容 payload 时必须显示明确错误，不得退回截图。  
   **来源**：2026-08-29 Export 方案 A 补充约束。
5. **R5 — `book.app.api` 平台策略**：macOS 不做 ShapeRange/appscript 适配，直接走 `sheet.pictures` 遍历 fallback；Windows 保留现有 `Selection.ShapeRange` 路径。  
   **来源**：2026-08-29 用户访谈决策 3。
6. **R6 — 范围选择替代 UI**：`_select_sample_data()` 在 macOS 不调用 `book.app.api.InputBox(Type=8)`，改用 tkinter 输入活动工作表范围地址；Windows 继续使用现有 InputBox。  
   **来源**：2026-08-29 用户访谈决策 3。
7. **R7 — 平台边界**：严格排除 WPS for Mac；本 PR 只支持 Microsoft Excel for Mac。  
   **来源**：2026-08-29 用户访谈决策 4。
8. **R8 — 最低环境**：声明支持 macOS 10.14+、Excel for Mac 2016+、Intel 与 Apple Silicon；Python 延续项目现有最低版本 3.10。  
   **来源**：2026-08-29 用户访谈决策 5；`pyproject.toml:9-22`。
9. **R9 — 验收结构**：以 CI 可执行验证为主，包括完整单元测试、新增平台分支覆盖、静态检查；Excel for Mac 实机验收由用户执行并记录。  
   **来源**：2026-08-29 用户访谈决策 6。
10. **R10 — Windows 零行为改动**：只允许平台分支与新增能力；Windows 原有逻辑保持不变，所有现有 `ribbon/*.bas` 文件零修改。payload 写入在 Windows 上只能是不会改变现有结果的新增 best-effort 副作用。  
    **来源**：2026-08-29 用户访谈决策 7；Export 方案 A 补充约束。
11. **R11 — payload 模块边界**：新建独立模块 `xstars/artifacts.py`；生成链路的登记调用可平台无关，但平台导出分发必须显式隔离。  
    **来源**：2026-08-29 Export 方案 A 补充约束。
12. **R12 — RunPython 复用**：macOS 使用现有 `ribbon/ribbon_callbacks.bas` 的 RunPython 回调，不新建或修改 AppleScriptTask VBA；安装文档说明 xlwings 所需安装步骤。  
    **来源**：2026-08-29 用户访谈决策 1、7；本地外部调研简报对 xlwings RunPython 的结论。

### 期望行为

 1. **R13 — 可诊断性**：artifact 保存失败不弹出阻塞对话框，但应至少保留可供测试和故障定位的状态/日志；Export payload 缺失应说明需重新生成图表，而不是显示通用异常。
 2. **R14 — 安全序列化**：推荐采用带 schema version 的非可执行格式，不使用可触发任意代码执行的 pickle；加载时校验 schema、必要字段和图片关联键。
 3. **R15 — 文档可操作**：安装文档覆盖 Python/虚拟环境、`pip install -e ".[dev]"`、`xlwings addin install`、`xlwings runpython install`、导入现有 `ribbon_callbacks.bas`、macOS 自动化权限、启动与故障排查，并明确 `.app` 不在本 PR。

## 3. Non-goals

1. 不构建或分发 macOS 独立 `.app`、PyInstaller bundle、universal2 二进制、DMG 或 Homebrew 包。
2. 不实现 Developer ID 签名、Hardened Runtime、entitlements、Apple Events 权限打包或 Apple notarization。
3. 不新增 AppleScriptTask VBA、`MacScript`、`RunFrozenPython` 或 macOS standalone VBA；这些属于后续独立分发 PR。
4. 不支持 WPS for Mac，也不合并或依赖 `origin/feature/wps-support`。
5. 不迁移到 Office.js/Web Add-in，不引入本地 HTTP/FastAPI 服务。
6. 不支持导出任意 Excel Shape、用户自建 Excel 图表、Range 截图或剪贴板图片。
7. 不为 macOS 适配 `Selection.ShapeRange`，不引入 appscript ShapeRange 专用实现。
8. 不改变 Windows 的 COM `CopyPicture` + `PIL.ImageGrab` 导出路径、InputBox(Type=8) 流程、Ribbon 回调或安装包模式。
9. 不修改 `ribbon/ribbon_callbacks.bas`、`ribbon/ribbon_callbacks_installed.bas`、`ribbon/ribbon_callbacks_standalone.bas` 或其他现有 `.bas` 文件。
10. 不承诺 Excel for Mac 2011、macOS 10.13 及更早版本、Python 3.9 及更早版本。
11. 不引入 xlwings UDF；XSTARS 当前不依赖 UDF。
12. 不在本 PR 中重构整个 `xstars/main.py`、统计核心、绘图核心、配置持久化或宿主适配架构。
13. 不保证旧版本 XSTARS 已插入、但从未生成 artifact payload 的图表可以在 macOS 导出；应提示重新生成。

## 4. Research summary

### 4.1 代码库现状与可复用模式

| 结论 | 证据 | 计划含义 |
| --- | --- | --- |
| XSTARS 的主路径是 VBA RunPython → xlwings → pandas/scipy/matplotlib → `sheet.pictures.add(fig)`。 | `ribbon/ribbon_callbacks.bas:1-128`；`xstars/main.py:205-867`；Explore 报告 §1。 | macOS MVP 可复用现有 VBA 与 Python 核心，只隔离直接 COM 调用。 |
| 当前图片选择先访问 COM `Selection.ShapeRange`，异常后遍历活动工作表 `pictures`，且 fallback 只筛选 `XSTARS_Plot*`。 | `xstars/main.py:881-894` `_get_selected_shapes()`。 | Darwin 分支应跳过 `.api.Selection.ShapeRange`；可复用 pictures 遍历思路，但“可导出图表集合”须由 artifact 登记定义。 |
| 当前高清导出依赖 COM Shape 属性、`CopyPicture(1, 2)`、剪贴板等待和 `PIL.ImageGrab.grabclipboard()`。 | `xstars/main.py:897-942` `_export_shape_highres()`。 | 这是 macOS blocker；Windows 函数保持原样，另建 Darwin artifact 导出路径。 |
| Export 是独立 RunPython 调用；当前只持有 Excel Shape，不持有原 Figure、处理后数据或绘图配置。 | `xstars/main.py:870-878`、`1066-1092`；生成链路见 `271-423`、`436-714`、`744-867`、`1255-1382`。 | 必须在生成时持久化重建 payload，不能假设 Export 调用仍有内存中的 Figure。 |
| `PlotEngine.plot(df_wide, stats_result)` 返回 Figure；`export_figure()` 已以 `fig.savefig(path, dpi, bbox_inches="tight")` 导出。 | `xstars/plot_engine.py:21-23` 及 `PlotEngine.plot()`。 | artifact 重建应复用 PlotEngine/现有专用渲染逻辑与 `export_figure()`，避免复制格式转换代码。 |
| 主流程、Quick、预设、WB/qPCR labeled、ELISA、标准曲线在多个位置生成并插入 Figure。 | `xstars/main.py:271-423`、`436-714`、`744-867`、`1255-1382`。 | artifact 登记必须覆盖被声明为“可导出”的每种 XSTARS 图表生成点，不能只改普通 Run。 |
| `PrismConfig.to_dict()` 会丢弃 export、control、preset 与若干 transient 字段。 | `xstars/config.py:145-191`。 | artifact 需要专用快照 DTO/serializer；不得直接改变用户设置文件的既有语义。 |
| `StatsResult`/`PairResult` 是绘图显著性标注所需的结构化结果。 | `xstars/stats_engine.py:21-78`。 | payload 需无损保存/恢复绘图所需统计字段，或保存足够信息以确定性重算。 |
| `_select_sample_data()` 当前调用 `book.app.api.InputBox(Type=8)`，取消和 API 失败都返回 `None`。 | `xstars/main.py:1385-1413`。 | Darwin 分支应使用 tkinter 地址输入并保留“取消返回 None”的上层契约。 |
| `_show_error()` 使用 tkinter `-topmost`，若整个 tkinter 块失败才尝试 VBA `MsgBox`。 | `xstars/main.py:166-199`；统一入口 `201-208` 及各 `run_*()` wrapper。 | 应把置顶失败降级为非致命，避免 macOS 仅因窗口属性差异丢失 tkinter 错误消息。 |
| 设置路径使用 `Path.home() / ".xstars" / "settings.json"`。 | `xstars/config.py:12`。 | `~/.xstars/artifacts/` 符合现有跨平台目录约定，无需改变 settings 路径。 |
| 字体链包含 Arial、Helvetica、DejaVu Sans。 | `xstars/styles.py:26`。 | 当前字体策略可复用；实机只验证回退效果，不预设代码修改。 |
| 测试通过 MagicMock 模拟 xlwings，无需真实 Excel。 | `tests/test_end_to_end.py:32-61`；Explore 报告 §3。 | 平台分支和 artifact 生命周期可在 CI 中以 `sys.platform` patch 与 xlwings mock 覆盖。 |
| 项目没有已配置 linter；`pyproject.toml` 仅配置 pytest，dev 依赖为 pytest/pytest-cov。 | `pyproject.toml:24-31`。 | 本 PR 的无新增依赖静态门槛采用 `compileall` + `git diff --check`；是否引入 Ruff 留待后续工程化决策。 |
| 英文/中文 README 目前均把系统要求写为 Windows，Ribbon 文档只给通用安装步骤。 | `README.md` Quick Start/Requirements；`README.zh-CN.md` 快速开始/系统要求；`ribbon/README.md:1-19`。 | 需要增加 macOS 开发者模式说明，同时避免暗示 macOS 有独立安装包。 |

### 4.2 项目策略与候选方案取舍

- Explore 报告记录：`origin/feature/wps-support`（当时 SHA `59c225a`）中的 `docs/cross-platform-office-technology-strategy.md` 已将 **Excel macOS = VBA + xlwings** 定为推荐路线，并否决在该阶段迁移 Office.js；该策略文档尚不在主分支。证据为 `plans/explore-20260829-macos-compat.md` §6b 的分支差异摘要，属于间接读取证据。
- 候选路径中采用“开发者模式 RunPython + Python 外科式平台分支”的 MVP；拒绝本 PR 内的 `.app`/MacScript/AppleScriptTask standalone 路径，原因是后者引入沙盒、签名、公证及分发基础设施，超出已批准范围。
- Export 采用用户批准的方案 A：生成时保存重建上下文，导出时重建 Figure。相较 AppleScript/剪贴板方案，该方案不依赖不稳定的 Excel for Mac 图形导出 API，并能复用现有 Matplotlib 渲染结果。

### 4.3 外部调研结论

> 以下仅复用本地调研简报，不进行网络请求。日期为简报记录的文档年份/更新时间；未记录具体日期处明确标注。

| 外部结论 | 来源、日期、置信度 | 采用方式 |
| --- | --- | --- |
| xlwings 支持 macOS 10.14+、Excel for Mac 2016+、Intel 与 Apple Silicon。 | [xlwings Installation](https://docs.xlwings.org/en/stable/installation.html)，简报标注 2025，置信度高。 | 作为声明的最低支持矩阵；仍需实机记录。 |
| Windows `.api` 是 COM，macOS `.api` 是 appscript reference，COM 成员不能假设可用。 | [xlwings App API](https://docs.xlwings.org/en/stable/api/app.html)，简报标注 2024，置信度高。 | 所有现有直接 COM 调用必须受 Darwin 分支隔离。 |
| `sheet.pictures.add(fig)` 在 macOS 可用，xlwings 会经沙盒临时目录插入 Matplotlib Figure。 | [xlwings `_xlmac.py`](https://github.com/xlwings/xlwings/blob/main/xlwings/_xlmac.py)，简报标注 2024，置信度高。 | 保留当前插图主路径，并纳入实机检查。 |
| Excel for Mac 缺少稳定的 COM 等价 Shape/Chart 图片导出接口；AppleScript 历史导出命令可能报 `-50` 或受沙盒路径限制。 | [Apple StackExchange: Save Excel Chart as PNG](https://apple.stackexchange.com/questions/318087/applescript-to-save-excel-chart-as-png-broken-after-recent-update)，简报未记录发布日期，置信度中高。 | 不实现 Excel 反向导出；选择 Python 重建。 |
| xlwings RunPython 在 Mac 可用，其 Mac 路径使用 AppleScriptTask；需安装 `xlwings.applescript`。 | [xlwings CLI](https://docs.xlwings.org/en/stable/command_line.html)，简报标注 2024，置信度高。 | 文档要求 `xlwings runpython install`，并复用现有 RunPython VBA。 |
| `xlwings addin install` 支持 macOS，并安装 add-in 到 Office Group Container 启动目录。 | [xlwings Add-in & Settings](https://docs.xlwings.org/en/stable/addin.html)，简报标注 2024，置信度高。 | 纳入开发者安装步骤和故障排查。 |
| `MacScript` 已弃用，沙盒下官方推荐 AppleScriptTask。 | [Microsoft Learn: AppleScriptTask](https://learn.microsoft.com/en-us/office/vba/office-mac/applescripttask)，简报标注 2023 更新，置信度高。 | 仅作为后续独立分发背景；本 PR 不新增脚本桥。 |
| macOS 独立分发涉及签名、公证、Hardened Runtime 与 Apple Events entitlement。 | [PyInstaller macOS signing/notarization](https://pyinstaller.org/en/stable/feature-notes.html#macos-binary-signing-and-notarization)，简报标注 2024，置信度高；[Apple notarization](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution)，简报未记录日期，置信度高。 | 证明 `.app` 必须留到独立后续 PR。 |

### 4.4 推荐方案

1. 新建 `xstars/artifacts.py`，提供 versioned payload DTO、保存、加载、删除/校验、Figure 重建接口；默认根目录建议 `~/.xstars/artifacts/`。
2. payload 推荐使用非可执行、可校验的 JSON（必要数组用 JSON-safe 表达），显式保存 DataFrame 列顺序/缺失值、完整绘图配置快照、StatsResult/PairResult、renderer kind 及专用参数；禁止 pickle。
3. artifact key 推荐由工作簿标识、工作表名、Excel picture name 组成并做路径安全哈希；manifest 保留可诊断元数据。保存采用临时文件 + 原子替换，并捕获全部文件系统/序列化错误，确保 best-effort。
4. 所有合格图表在 `sheet.pictures.add()` 成功并取得最终 picture name 后登记 payload。特殊图（如标准曲线）使用 renderer kind 和专用参数，而不是强行套用普通 `PlotEngine.plot()`。
5. `_run_export_impl()` 显式按 `sys.platform == "darwin"` 分发：Darwin 仅接受 artifact-backed XSTARS pictures；其他平台继续调用原 `_export_shape_highres()`。
6. `_select_sample_data()` 仅在 Darwin 使用 tkinter 地址输入；Windows 源路径保持不变。
7. 以新增单元测试和 macOS/Windows CI matrix 锁定分支，实机验证 Excel 桥、窗口焦点和系统权限。

### 4.5 替代方案与不采用原因

- **AppleScript/appscript ShapeRange 适配**：用户已明确否决；行为跨 Excel 版本不稳定。
- **剪贴板/PyObjC/pngpaste**：会重新引入系统级截图、额外依赖与权限，不符合 fig.savefig 决策。
- **只在生成对话框即时导出**：不能保留现有 Ribbon Export 用户流程，且用户已批准跨调用 payload。
- **pickle Figure/对象**：存在安全与版本兼容风险，不推荐。
- **Office.js + 本地服务**：架构和部署面过大，属于 Non-goal。
- **新建 macOS `.bas`**：开发者模式可直接复用现有 RunPython 文件，且回归约束禁止修改现有 VBA。

### 4.6 证据缺口

1. 尚未在 Excel for Mac 实机验证 `InputBox(Type=8)`、`ShapeRange`、`sheet.pictures` 遍历和 tkinter 焦点行为；前两项已通过产品决策绕开，但仍需验证 fallback 与 UI。
2. CI 的现代 macOS runner 无法证明 macOS 10.14 下真实 Excel/GUI 行为；最低版本声明主要基于 xlwings 文档，最终由用户拥有的目标设备记录覆盖范围。
3. 当前没有可直接复用的 Figure/payload schema，也没有跨工作簿 Save As/重命名后的 artifact 身份策略；须在 T1.1 锁定。
4. `PrismConfig` 有 transient/non-persisted 字段，普通图和专用曲线需要不同重建数据；需用 round-trip 测试证明视觉语义所需字段齐全。
5. 本地调研简报未记录所有网页的精确发布日期/访问日期；本计划不在无网络约束下补猜日期。
6. 策略文档只通过 Explore 报告的分支摘要间接核对，主分支没有该文件；本 PR 不依赖其合入。

## 5. Gap analysis

| ID | 功能缺口 | 现状 | 影响 | 补齐任务 |
| --- | --- | --- | --- | --- |
| G1 | 缺少跨 RunPython 调用的 Figure 重建上下文 | 图片插入后只剩 Excel Shape；Export 调用没有原 Figure、数据、配置或统计结果 | macOS 无法按批准方案用 fig.savefig 导出已有图表 | T1.1、T1.2、T1.3、T3.1 |
| G2 | 当前 Export 是 Windows COM/剪贴板实现 | `_export_shape_highres()` 使用 Shape 属性、CopyPicture、ImageGrab | macOS Export blocker；直接替换会破坏 Windows | T2.1、T2.2、T3.2、T3.3 |
| G3 | Shape 选择直接触达 COM ShapeRange | `_get_selected_shapes()` 先调用 `book.app.api.Selection.ShapeRange` | macOS appscript 对象不保证该 COM 接口 | T2.1、T3.2 |
| G4 | 标准曲线/ELISA 第二选区依赖 COM InputBox(Type=8) | `_select_sample_data()` 无平台分支，API 异常与取消均返回 None | macOS 用户无法可靠选取样本范围，且错误不可区分 | T2.3、T3.2、T5.2 |
| G5 | tkinter 错误窗的 `-topmost` 失败会放弃整个 tkinter 路径 | `_show_error()` 把 root 创建、置顶、messagebox 放在同一 try 中 | macOS 窗口层级差异可能让错误消息不可见 | T2.4、T3.2、T5.2 |
| G6 | macOS 开发者安装与权限说明缺失 | README 只声明 Windows；Ribbon 文档未列 Mac RunPython 脚本安装 | 用户无法复现受支持环境，容易误认为有 `.app` | T4.1、T4.2、T4.3 |
| G7 | 缺少平台分支 CI 与静态门禁 | 现有 pytest 可 mock Excel，但无 macOS/Windows matrix；无 linter 配置 | 无法持续证明 Darwin 分支与 Windows 零回归 | T3.1、T3.2、T3.3、T5.1、T5.3 |
| G8 | 缺少 Excel for Mac 实机证据 | 自动化测试不启动 Excel、VBA、tkinter 或系统权限弹窗 | CI 通过仍不能证明端到端可用 | T5.2、T5.3 |
| G9 | 缺少 Windows 变更边界的机器可判定保护 | 需求明确 `.bas` 零修改和 Windows 逻辑原样，但当前无专门断言 | 平台改造可能误改 Windows Shape/InputBox/Ribbon 行为 | T2.1、T2.2、T2.3、T3.3、T5.1、T5.3 |
| G10 | 本地 artifact 的损坏、陈旧、隐私和清理行为未定义 | `~/.xstars/` 目前只有 settings；没有 schema/version/retention | 导出可能误匹配或泄露实验数据，磁盘失败可能影响体验 | T1.1、T1.3、T3.1、T4.1 |

**Gap 自查**：G1–G10 均至少映射到一个稳定 To-do ID；所有 To-do 均支撑上述缺口或 R9/R10 横切约束，**无孤立缺口**。

## 6. Milestone 表格

| Milestone | Status | Dependencies | Validation | Notes |
| --- | --- | --- | --- | --- |
| M1 — Artifact 契约与生成时登记 | [x] | Plan 批准；待决事项 D1–D4 定案 | artifact round-trip、schema 拒绝、原子写入/best-effort、Figure 重建单测通过 | 先解决跨调用数据基础；不得阻断原出图 |
| M2 — macOS Python 平台分支 | [x] | M1 | Darwin Export、pictures fallback、tkinter 范围输入、错误窗降级测试通过；Windows 路径断言通过 | 不修改 VBA；COM 函数保留 |
| M3 — 回归与平台测试 | [x] | M1、M2 | 新增测试文件通过，完整 `pytest` 通过，Windows mock 行为与 `.bas` 零 diff | 测试采用 mock，不需要 Excel |
| M4 — macOS 开发者文档 | [x] | M2；安装命令已由调研确认 | 文档链接有效、步骤和 Non-goals 一致、英文/中文入口可发现 | 明确仅开发者模式 |
| M5 — CI、静态检查与实机验收 | [ ] | M3、M4 | macOS/Windows CI 全绿；`compileall`、`git diff --check` 通过；用户提交实机清单 | 用户实机记录前保持 Draft |

Milestone 总数：**5**（≤7，且 ≤10）。所有初始 Status 均为 `[ ]`。

## 7. 分 milestone 的 To-do checkbox 清单

### M1 — Artifact 契约与生成时登记

- [x] T1.1 定义并实现 versioned artifact payload 与安全持久化契约
  - 文件：新建 `xstars/artifacts.py`
  - 修改：定义 artifact key、schema version、renderer kind、DataFrame/config/stats/专用参数的 JSON-safe DTO；实现路径安全、schema 校验、临时文件原子替换、load/save 与清晰异常；默认根目录为 `~/.xstars/artifacts/`；禁止 pickle。
  - 验收：普通图 payload round-trip 后列顺序、NaN、枚举配置、统计 pairs 均等价；未知 schema、缺字段、损坏 JSON 被拒绝并返回可诊断错误；写入测试可注入临时根目录。
  - 依赖：Plan 批准；D1–D4 定案；支撑 G1、G10、R3、R4、R11、R14。

- [x] T1.2 为所有本 PR 声明支持导出的 XSTARS 图表生成点登记可重建 payload
  - 文件：修改 `xstars/main.py`；调用新建 `xstars/artifacts.py`
  - 修改：在普通 Run、Quick、预设、WB/qPCR labeled、ELISA/标准曲线等合格生成路径中，以最终 picture name 关联处理后数据、实际 plot config、StatsResult 或专用曲线参数；仅在 `sheet.pictures.add()` 成功后登记；提取最小 helper，避免散落重复逻辑。
  - 验收：每个已纳入的 renderer kind 都能生成 artifact，并可重建 Figure；payload key 与 sheet/picture 一致；登记失败不改变 Excel 插图、统计输出或 status bar 成功结果。
  - 依赖：T1.1；支撑 G1、G10、R2–R4。

- [x] T1.3 实现 best-effort、缺失/陈旧/损坏 artifact 的用户可诊断行为
  - 文件：修改 `xstars/main.py`；新建/修改 `xstars/artifacts.py`
  - 修改：生成时捕获 artifact 文件系统/序列化错误并记录非阻塞诊断；加载时区分 missing、corrupt、unsupported schema、renderer unsupported；为上层提供“重新生成图表”的友好消息。
  - 验收：模拟 PermissionError、磁盘写失败、损坏文件时生成流程仍成功；Export 不产生半成品输出，错误文本说明原因和恢复操作。
  - 依赖：T1.1、T1.2；支撑 G1、G10、R4、R13。

### M2 — macOS Python 平台分支

- [x] T2.1 将图表发现逻辑按平台隔离
  - 文件：修改 `xstars/main.py:881-894`
  - 修改：Darwin 直接遍历活动 sheet 的 pictures，并仅返回能关联有效 artifact 的 XSTARS 图片候选；不访问 `book.app.api.Selection.ShapeRange`。非 Darwin 保留现有 ShapeRange-first 与 fallback 行为。
  - 验收：Darwin mock 断言 `.app.api` 未被访问；只返回合格 XSTARS picture；非 Darwin 测试断言 ShapeRange 访问次数、返回顺序和 fallback 与基线一致。
  - 依赖：T1.1、T1.2；支撑 G2、G3、G9、R5、R10。

- [x] T2.2 新增 macOS artifact Figure 导出分支并保留 Windows COM 实现
  - 文件：修改 `xstars/main.py:897-1092`；调用 `xstars/artifacts.py` 与现有 `xstars/plot_engine.py:21-23`
  - 修改：保留 `_export_shape_highres()` 的 Windows COM/clipboard 代码；为 Darwin 新增 load payload → rebuild Figure → `export_figure()`/`fig.savefig()` → close Figure 流程；多图命名继续沿用现有 `_1`、`_2` 规则；缺失 payload 调用统一友好错误。
  - 验收：Darwin 测试中 `CopyPicture`、ImageGrab 和 COM Shape 属性均未调用；PNG/TIFF/JPG/SVG/PDF 中现有 Export 对话框声明格式按 Matplotlib 能力输出；DPI 与多图文件名正确；非 Darwin `_export_shape_highres()` 调用契约不变。
  - 依赖：T1.1–T1.3、T2.1；支撑 G1、G2、G9、R2–R5、R10。

- [x] T2.3 为 `_select_sample_data()` 增加 Darwin tkinter 范围地址输入
  - 文件：修改 `xstars/main.py:1385-1413`
  - 修改：Darwin 用 tkinter `simpledialog`/等价小型输入 UI 请求活动 sheet A1 范围，使用 xlwings `sheet.range(address)` 读取；保留 DataFrame 清洗；取消返回 `None`；无效地址给出可理解反馈。非 Darwin 保留现有 `book.app.api.InputBox(Type=8)`。
  - 验收：Darwin 有效地址产生与 Windows 路径同形 DataFrame；取消返回 None；无效地址不访问 COM 且不泄露 traceback；非 Darwin mock 断言原 InputBox 调用参数不变。
  - 依赖：D5 定案；支撑 G4、G9、R6、R10。

- [x] T2.4 使 tkinter `-topmost` 失败成为非致命降级
  - 文件：修改 `xstars/main.py:166-199`
  - 修改：把 `root.attributes("-topmost", True)` 置于局部容错中，确保该属性失败时仍尝试 messagebox；所有路径可靠销毁 root；不改变错误文案与 Excel status bar。
  - 验收：模拟 `attributes` 抛错时 `messagebox.showerror` 仍调用且 root.destroy 执行；模拟 tkinter 整体失败时保留现有 VBA MsgBox/status bar fallback。
  - 依赖：无；支撑 G5、R13。

### M3 — 回归与平台测试

- [x] T3.1 增加 artifact schema、round-trip、原子写入与失败隔离单元测试
  - 文件：新建 `tests/test_artifacts.py`
  - 修改：覆盖普通/专用 renderer payload、DataFrame 缺失值、config/stats 恢复、临时目录、schema version、损坏/缺失文件、PermissionError、并发/半文件可见性最低保障。
  - 验收：测试不读写真实 `~/.xstars`；失败场景可重复；重建 Figure 可由 Agg backend 导出非空文件。
  - 依赖：T1.1–T1.3；支撑 G1、G7、G10、R9。

- [x] T3.2 增加 Darwin 平台分支单元测试
  - 文件：新建 `tests/test_macos_support.py`；复用 `tests/test_end_to_end.py:32-61` mock 模式
  - 修改：patch 平台与 xlwings/tkinter；覆盖 pictures fallback、artifact-only 过滤、重建导出、多图命名、payload 错误、范围输入有效/取消/非法、topmost 降级。
  - 验收：测试全程不启动 Excel、不请求 GUI、不依赖本机用户目录；每个新 Darwin 分支至少有成功与失败用例。
  - 依赖：T2.1–T2.4；支撑 G2–G8、R9。

- [x] T3.3 增加 Windows 零行为改动回归断言
  - 文件：修改/新增 `tests/test_macos_support.py`；必要时仅追加 `tests/test_end_to_end.py`
  - 修改：patch 非 Darwin 平台，锁定 ShapeRange-first、现有 pictures fallback、COM high-res helper 调用、InputBox(Type=8) 参数及 RunPython 主入口；加入 `.bas` diff 作为 CI 门禁而非改写 fixture。
  - 验收：Windows mock 断言与改造前调用契约一致；所有现有测试不需改期望即可通过；`git diff origin/main...HEAD -- ribbon/*.bas` 为空。
  - 依赖：T2.1–T2.3；支撑 G2、G7、G9、R10。

### M4 — macOS 开发者文档

- [x] T4.1 新增 macOS 开发者安装与故障排查文档
  - 文件：新建 `docs/macos-developer-setup.md`
  - 修改：写明支持矩阵、Python 3.10+、虚拟环境、editable install、`xlwings addin install`、`xlwings runpython install`、复用 `ribbon/ribbon_callbacks.bas`、Excel 宏设置、自动化权限、首次运行、artifact 目录/实验数据隐私/清理、Export 限制与常见错误。
  - 验收：一名未参与实现的开发者可按步骤完成安装；明确不提供 `.app`、不支持 WPS、旧图 payload 缺失需重新生成。
  - 依赖：T1.1–T2.4；支撑 G6、G10、R1、R7、R8、R12、R15。

- [x] T4.2 更新英文与中文入口文档的平台声明
  - 文件：修改 `README.md`、`README.zh-CN.md`
  - 修改：在 Quick Start/快速开始与 Requirements/系统要求中区分 Windows installer 和 macOS developer mode；链接 macOS 文档；避免把“无需 Python”描述套用于 macOS。
  - 验收：中英文平台表述一致；所有链接为仓库相对链接；Non-goals 和最低版本无冲突。
  - 依赖：T4.1；支撑 G6、R1、R7、R8。

- [x] T4.3 更新 Ribbon 文档以明确复用现有 RunPython VBA
  - 文件：修改 `ribbon/README.md`
  - 修改：增加 macOS 小节，说明导入现有 `ribbon_callbacks.bas`，列出 xlwings add-in/runpython 前置条件和 Mac VBA Editor 操作差异；不得修改 `.bas`。
  - 验收：文档中不出现新增 macOS `.bas` 或 Shell/AppleScriptTask 实现要求；与 `docs/macos-developer-setup.md` 互链。
  - 依赖：T4.1；支撑 G6、R10、R12。

### M5 — CI、静态检查与实机验收

- [ ] T5.1 新增 macOS/Windows 自动化验证工作流
  - 文件：新建 `.github/workflows/macos-support.yml`
  - 修改：在 `macos-latest` 与 `windows-latest`、Python 3.10 上安装 `.[dev]`，执行专项目标测试、完整 pytest、`compileall`；增加 `ribbon/*.bas` 相对基线零 diff 检查（可在独立 Linux job 执行）。
  - 验收：PR workflow 两个平台全绿；无需 Excel/GUI；任何 `.bas` 变更使门禁失败；不新增生产依赖。
  - 依赖：T3.1–T3.3；支撑 G7、G9、R9、R10。

- [ ] T5.2 提供并执行 Excel for Mac 实机验收清单
  - 文件：新建 `docs/macos-manual-acceptance.md`；用户在 Draft PR 描述/评论记录结果
  - 修改：列出安装、Ribbon Run/Quick/预设、picture 插入、普通与专用图 artifact、PNG/TIFF/SVG/PDF 导出、DPI、payload 缺失提示、标准曲线/ELISA 地址输入、取消/非法范围、错误窗、设置持久化、Intel/Apple Silicon 与权限信息；明确责任人为用户。
  - 验收：用户逐项记录日期、macOS/Excel/Python/xlwings 版本、芯片、结果和截图/文件证据；所有 blocker 项通过或有获批豁免。
  - 依赖：T2.1–T4.3；支撑 G4–G8、R8、R9。

- [ ] T5.3 执行最终范围、回归和发布前审查
  - 文件：仅审查本计划“文件级修改范围”所列文件；不新增实现范围
  - 修改：运行 Validation contract 全部自动命令；核对完整 diff、无 `.bas` 修改、无源码外意外文件、无 staged 临时产物；把用户实机结果回填 PR。
  - 验收：所有自动门禁通过；reviewer 无 blocker；用户实机清单完成；Draft 才可转 Ready；Gap/任务映射保持无孤立缺口。
  - 依赖：T5.1、T5.2；支撑 G7–G9、R9、R10。

## 8. Validation contract

### 8.1 自动化检查

| 检查项 | 命令或验证方式 | 预期结果 | 通过标准 | 责任人/限制 |
| --- | --- | --- | --- | --- |
| Artifact 专项测试 | `python -m pytest tests/test_artifacts.py -v` | round-trip、schema、失败隔离、Figure 重建全部通过 | 0 failed；不写真实用户目录 | 实施者；CI 可执行 |
| macOS 分支专项测试 | `python -m pytest tests/test_macos_support.py -v` | Darwin 与 Windows 模拟分支全部通过 | 0 failed；测试不启动 Excel/tkinter GUI | 实施者；CI 可执行 |
| 完整回归 | `python -m pytest` | 现有与新增测试全部通过 | 命令退出码 0，0 failed/0 errors | 实施者；CI 可执行 |
| 覆盖率观察 | `python -m pytest --cov=xstars --cov-report=term-missing` | 新增 artifact 和平台分支有可见覆盖 | 新增分支均被成功/失败测试触发；本 PR不擅自设全仓阈值 | 实施者；现有 dev 依赖含 pytest-cov |
| 静态语法检查 | `python -m compileall -q xstars tests` | 无 SyntaxError/编译错误 | 退出码 0 | 实施者；当前仓库无 linter 配置 |
| 格式/空白错误 | `git diff --check` | 无 trailing whitespace/conflict marker | 无输出，退出码 0 | 实施者 |
| VBA 零修改 | `git diff --exit-code origin/main...HEAD -- 'ribbon/*.bas'` | 所有现有 `.bas` 与基线相同 | 无 diff，退出码 0 | 实施者；若基线分支名不同，替换为实际 PR base |
| 变更范围 | `git diff --name-status origin/main...HEAD` | 只出现第 9 节允许文件 | 无未批准文件、无删除 | 实施者 + reviewer |
| Artifact 安全 | 代码审查 + 单测：搜索 `pickle`/任意执行入口，验证路径哈希与 schema | 不加载可执行对象，不接受目录穿越 | reviewer 无安全 blocker | reviewer required |
| Windows 行为 | Windows CI + 非 Darwin mock | COM/ShapeRange/InputBox 路径调用契约不变 | Windows job 全绿；相关断言全过 | CI + reviewer |
| macOS 无 COM 导出 | Darwin mock 断言 | 不调用 ShapeRange/CopyPicture/ImageGrab | 专项测试全过 | CI |
| CI matrix | GitHub Actions `macos-latest`、`windows-latest` / Python 3.10 | 安装与测试成功 | 所有 required jobs green | CI；不代表真实 Excel GUI |
| 无 staged 临时文件 | `git diff --cached --exit-code`（在交付工作树执行）及 `git status --short` 人工审阅 | 无意外 staged/未跟踪生成物 | 无 artifact、图片、缓存、临时 payload 进入提交 | 实施者 |

### 8.2 功能行为断言

1. **生成成功 + artifact 成功**：Excel 图表行为与基线一致；本地出现与 workbook/sheet/picture 对应的有效 payload；重建 Figure 可导出非空文件。
2. **生成成功 + artifact 写失败**：统计表、处理后数据、Excel picture 与成功 status 不受阻断；诊断可定位写入失败；后续 macOS Export 明确提示重新生成/检查目录权限。
3. **macOS Export 成功**：仅对 artifact-backed XSTARS picture 生效；使用 payload 重建；所选 DPI 与扩展名生效；不得调用 Excel 反向截图。
4. **macOS Export 失败**：非 XSTARS Shape、legacy picture、missing/corrupt/version-mismatch payload 都不会截图或崩溃；显示原因和恢复动作。
5. **Windows Export**：仍以原 COM CopyPicture + ImageGrab 路径执行；新增 artifact 不改变输出路径、命名、DPI 或 Shape 选择。
6. **范围输入**：Darwin 接受活动 sheet 的约定 A1 地址并返回清洗后的 DataFrame；取消返回 None；非法值有反馈；Windows 仍调用 Type=8 InputBox。
7. **错误显示**：`-topmost` 不支持时仍显示 tkinter 错误；tkinter 整体不可用时仍保留现有 fallback。

### 8.3 Excel for Mac 人工验收

责任人：**用户**。实施者负责提供 `docs/macos-manual-acceptance.md`，用户负责在真实 Excel for Mac 中执行并记录。

必须记录：

- 日期、芯片（Intel/Apple Silicon）、macOS、Excel、Python、xlwings 版本；
- `xlwings addin install`、`xlwings runpython install` 与现有 `ribbon_callbacks.bas` 的安装结果；
- Run、Quick Run、至少一个 preset、标准曲线/ELISA 范围输入；
- Matplotlib picture 插入、状态栏、设置保存；
- artifact 创建、正常导出、legacy/missing payload 报错；
- PNG、TIFF、SVG、PDF 与至少两个 DPI；
- tkinter 焦点、取消、非法范围、系统自动化权限提示；
- blocker 失败的截图/日志和复现步骤。

通过标准：所有 blocker 项通过；非 blocker 差异必须写入风险并获用户明确接受。无法由 CI 验证的原因：CI 不含 Excel GUI、VBA 宏信任、Apple Events 权限交互或目标用户的 tkinter 窗口管理环境。

### 8.4 Reviewer gate

- Reviewer 必须核对：Windows 分支是否原样保留、artifact 是否 best-effort、安全 schema 是否拒绝不可信输入、所有生成点是否有对应 payload、macOS 是否完全避开截图、文档是否未暗示 `.app` 支持。
- Blocker 规则：任一 `.bas` 变化、任一 Windows 行为变化、artifact 写失败阻断出图、macOS fallback 到截图、非 XSTARS Shape 可导出、缺少用户实机记录，均阻止合并。

## 9. 文件级修改范围 + 风险 / 回滚 / 待决事项 + Git 策略

### 9.1 文件级修改范围

| 文件 | 动作 | 允许内容 | 明确禁止 |
| --- | --- | --- | --- |
| `xstars/artifacts.py` | 新建 | payload DTO/schema、key/path、save/load/validate、Figure rebuild、artifact 错误类型 | pickle、任意代码执行、Excel COM、网络传输 |
| `xstars/main.py` | 修改 | artifact 登记 helper；Darwin pictures/export/InputBox 分支；topmost 局部降级 | 重构统计核心；改变非 Darwin COM/InputBox 行为 |
| `tests/test_artifacts.py` | 新建 | artifact round-trip、安全、失败隔离、重建测试 | 访问真实用户目录或 Excel |
| `tests/test_macos_support.py` | 新建 | Darwin/Windows 分支与 tkinter/xlwings mocks | 实机 GUI 依赖 |
| `tests/test_end_to_end.py` | 可选修改 | 仅复用/追加 Windows 零回归断言；若新文件足够则不改 | 改写现有测试期望以掩盖回归 |
| `docs/macos-developer-setup.md` | 新建 | macOS 开发者安装、权限、限制、artifact 隐私/清理、故障排查 | `.app` 已支持的表述 |
| `docs/macos-manual-acceptance.md` | 新建 | 用户实机 checklist 与记录模板 | 伪造实机结果 |
| `README.md` | 修改 | macOS developer mode 入口与平台条件 | 改变 Windows installer 承诺 |
| `README.zh-CN.md` | 修改 | 与英文一致的 macOS 开发者模式入口 | 扩大到 WPS Mac |
| `ribbon/README.md` | 修改 | 说明 Mac 复用现有 RunPython VBA | 要求新建/修改 `.bas` |
| `.github/workflows/macos-support.yml` | 新建 | macOS/Windows Python 3.10 测试、compileall、VBA diff gate | 打包、签名、公证、发布 job |
| `docs/cross-platform-office-technology-strategy.md` | 新建（D8 已批准修订） | 自 `origin/feature/wps-support` 复制并添加前言（说明本文档随本 PR 首次进入主线、WPS 适配为独立 PR） | 修改正文结论；替代 WPS PR 合入 |
| `xstars/tools/standard_curve.py` | 修改（2026-08-30 批准范围扩展） | 仅修复 `back_calculate` 对近零 OD 反算被掩码为 NaN 的预先存在缺陷，使 `TestZeroConcentration::test_back_calculate_includes_zero_range` 通过 | 改变拟合方法或统计语义；重构其他逻辑 |

**明确不修改**：

- `ribbon/ribbon_callbacks.bas`
- `ribbon/ribbon_callbacks_installed.bas`
- `ribbon/ribbon_callbacks_standalone.bas`
- 其他所有 `ribbon/*.bas`
- `xstars/plot_engine.py`（优先复用现有 `export_figure()`；若实施发现必须修改，须先更新 Plan 并重新批准）
- `xstars/config.py`、`xstars/styles.py`、统计/预设核心
- `xstars/cli.py`
- `pyproject.toml`、`requirements.txt`（本 PR 不新增生产或开发依赖）
- `XSTARS_Templates.xlsx`
- 安装器、PyInstaller、签名、公证相关文件

**不删除任何文件**。

### 9.2 风险

| 风险 | 等级 | 触发条件 | 缓解措施 |
| --- | --- | --- | --- |
| payload 含实验数据，造成隐私或残留风险 | 高 | 默认持久化处理后数据到用户目录；设备多人共享/备份同步 | 使用用户目录、限制文件权限、文档告知内容与清理命令；D3 确认 retention；不上传网络/提交仓库 |
| payload 不完整导致重建图与 Excel 中图片不一致 | 高 | 遗漏 transient config、StatsResult 或专用 renderer 参数 | versioned schema；每种 renderer round-trip；生成后测试重建；未知 renderer fail closed |
| Windows 意外回归 | 高 | 直接重写 `_export_shape_highres()`、ShapeRange 或 InputBox 公共路径 | 独立 Darwin dispatch；原函数保留；Windows CI、mock 断言、`.bas` diff gate |
| artifact 写失败阻断出图 | 高 | 权限、磁盘满、序列化异常 | 全部写操作 best-effort；临时文件原子替换；故障注入测试；成功主流程优先 |
| artifact 键碰撞或 Save As 后失联 | 中高 | 同名工作簿/sheet/picture，工作簿移动/重命名，未保存工作簿 | 安全哈希 + manifest；D2 明确身份规则；失配 fail closed 并提示重新生成，不误用其他数据 |
| stale/corrupt artifact 被错误加载 | 中 | 崩溃产生半文件、版本升级、手工修改 | schema/version/checksum/必要字段校验；原子写；损坏文件不导出 |
| tkinter 窗口焦点/深色模式差异 | 中 | Excel 前台、Tk 版本和 macOS WindowServer 行为不同 | `-topmost` 局部容错、用户实机焦点测试、状态栏/错误 fallback；不承诺原生 UI 观感 |
| CI 通过但真实 Excel 不工作 | 高 | mock 无法覆盖 VBA、Apple Events、Excel sandbox | 用户实机 checklist 为合并 blocker；记录明确版本/芯片/权限 |
| macOS 10.14 无可用 CI runner | 中 | GitHub hosted runner 仅提供现代 macOS | 文档依据官方支持范围；尽可能由用户旧系统验证；未验证版本明确记录 |
| 特殊图表生成点漏登记 | 中高 | 标准曲线/ELISA/WB/qPCR 与普通 PlotEngine 路径不同 | 生成点清单、renderer kind 测试、artifact-backed 过滤；未登记图明确提示而非截图 |
| 本地 artifacts 无限增长 | 中 | 长期生成大量图表且无 retention | D3 先确认；文档手动清理；后续可独立增加自动清理，不在未批准前猜测 |

### 9.3 回滚

1. **代码回滚**：按反向依赖顺序回滚 M5 → M4 → M3 → M2 → M1，或整体 revert 单个 feature PR。Darwin 支持将消失，Windows 因原路径保留应恢复到基线。
2. **CI 回滚**：删除新增 `.github/workflows/macos-support.yml`；不影响现有项目运行。
3. **文档回滚**：撤销 README/ribbon README 的 macOS入口，并删除两个新增 macOS 文档，避免继续宣称支持。
4. **本地数据清理**：用户可在退出 Excel/XSTARS 后删除 `~/.xstars/artifacts/`；不得删除 `~/.xstars/settings.json`，除非用户也要重置设置。
5. **数据兼容性**：artifact 是派生缓存，不是统计源数据或工作簿数据；删除后只影响 macOS Ribbon Export，重新生成图表可重建。旧 schema 应 fail closed，不执行迁移猜测。
6. **工作簿兼容性**：本方案不修改现有 VBA，不要求工作簿格式迁移。若 D2 最终批准把 artifact ID 写入 Excel picture metadata，则必须在实施前补充可逆清理步骤并重新审批；当前推荐不修改工作簿内容。
7. **部分回滚限制**：不得只回滚 artifact 写入而保留 Darwin Export，否则所有导出都会缺 payload；M2 依赖 M1，应整体回滚。

### 9.4 待决事项

以下事项未在原访谈中获得精确产品/API/架构批准；不得由实施者擅自拍板。Plan 获批时应逐项确认推荐值，或记录替代决定：

1. **D1 — payload 格式**：推荐带 `schema_version` 的 JSON、安全 DTO、禁止 pickle。需批准专用曲线参数的 JSON 表达与是否加入 checksum。
2. **D2 — artifact identity**：推荐 `hash(workbook full path or unsaved-book identity + sheet name + picture name)` + manifest，失配时 fail closed。需决定工作簿 Save As/移动后是自动重绑定、按内容查找，还是明确要求重新生成；推荐 MVP 要求重新生成以避免误匹配。
3. **D3 — retention/清理**：用户只批准存入本地目录，未指定保留期。推荐 MVP 不自动删除、文档提供按目录手动清理；后续 PR 再做 TTL/LRU。需确认隐私接受度。
4. **D4 — “XSTARS 生成图表”的精确定义**：推荐仅支持本版本生成且成功登记 payload 的所有 renderer kind；legacy 图或未登记的 XSTARS 图片提示重新生成。需确认标准曲线/ELISA fit curve 是否必须在 MVP 首批 renderer 集合中；计划当前按“全部当前 XSTARS 生成图”估算。
5. **D5 — tkinter 地址语法与重试**：推荐仅接受活动工作表 A1 地址（如 `A1:C6`），非法输入留在对话流程中重试，取消返回 None；不支持跨工作表/命名范围。需批准该交互细节。
6. **D6 — 静态检查工具**：仓库无 Ruff/mypy 配置。推荐本 PR 使用无新增依赖的 `compileall` + `git diff --check`，把 linter 引入留给独立工程化 PR；若用户要求 Ruff，必须批准修改 `pyproject.toml` 与 CI 范围。
7. **D7 — 最低版本实测覆盖**：需确认用户可提供哪些 Intel/Apple Silicon、macOS/Excel 组合。无法实测的组合只能标记为“基于上游支持声明”，不得写成已验证。
8. **D8 — 策略文档同步**：`docs/cross-platform-office-technology-strategy.md` 仅在 WPS 分支。推荐本 PR 不复制/修改该文件，只在 PR 描述说明独立关系；如需同步，必须另行批准以避免跨 PR 冲突。

### 9.5 Git 策略

- **分支名**：`feat/macos-support`
- **Draft PR 标题**：`feat: add macOS developer-mode support via xlwings RunPython`
- **PR 拆分决策**：一个 `feat/macos-support` 分支对应一个 Draft PR；M1–M5 串行进入同一 PR，不拆分。依据：Darwin Export 强依赖生成时 artifact，测试/文档/CI共同构成单一可验收 MVP；拆分会产生不可用中间态或暂时宣称支持但无法验证。
- **与 Windows + WPS PR 的关系**：本 PR 是独立 PR，不依赖 `origin/feature/wps-support` 合入；不复用 WPS JS bridge，不修改 WPS PoC。若两者同时触及 README/文档，合并时只解决文本冲突，不引入 WPS Mac 范围。
- **Milestone/提交建议顺序**：
  1. M1：`feat(artifacts): persist rebuild payloads for generated charts`
  2. M2：`feat(macos): add artifact export and range-input branches`
  3. M3：`test: cover macOS branches and Windows regression`
  4. M4：`docs: add macOS developer setup and acceptance guide`
  5. M5：`ci: validate macOS support and VBA immutability`
- **合并顺序**：M1 → M2 → M3 → M4 → M5 在分支内串行；Draft PR 只有在所有自动检查、reviewer gate 与用户实机 checklist 通过后才转 Ready 并 squash/rebase 合入目标主分支。WPS PR 可先或后独立合并；若存在冲突，后合并者 rebase 到最新主分支并重新运行完整 Validation contract。
- **Draft PR 描述草稿**：

```markdown
## Summary

Adds the first XSTARS macOS MVP for **developer mode only**:

- reuses the existing xlwings `RunPython` VBA callbacks
- adds Darwin-only Python branches for picture discovery and manual range input
- persists versioned rebuild payloads for XSTARS-generated charts
- rebuilds Matplotlib figures and exports them via `fig.savefig()` on macOS
- adds macOS developer setup, CI coverage, and a real-Excel manual checklist

## Scope

Supported target: macOS 10.14+, Excel for Mac 2016+, Python 3.10+, Intel and Apple Silicon.

This PR does **not** provide a standalone `.app`, PyInstaller packaging, signing/notarization, AppleScriptTask VBA, WPS for Mac, Office.js, or arbitrary Excel Shape export.

## Windows regression boundary

Windows behavior is intentionally unchanged:

- existing COM `ShapeRange` selection remains in use
- existing `CopyPicture` + `PIL.ImageGrab` export remains in use
- existing `InputBox(Type=8)` remains in use
- all existing `ribbon/*.bas` files remain byte-for-byte unchanged

Artifact persistence is a best-effort additive side effect and must never block chart generation.

## Export design

Each newly generated supported XSTARS picture gets a local, versioned rebuild payload under `~/.xstars/artifacts/`. On macOS, Export loads the payload, rebuilds the Matplotlib Figure, and saves it without reverse-exporting or screenshotting Excel. Missing, stale, or corrupt payloads fail with an explicit instruction to regenerate the chart.

## Validation

- [ ] artifact unit tests
- [ ] macOS platform-branch unit tests
- [ ] full pytest suite
- [ ] static syntax and diff checks
- [ ] macOS + Windows CI jobs
- [ ] `ribbon/*.bas` zero-diff gate
- [ ] Excel for Mac manual checklist completed by the user

## Relationship to Windows + WPS work

This is an independent PR and does not depend on `origin/feature/wps-support` being merged. It follows the documented architectural direction of keeping the Python core while using a host-specific bridge, but it does not include or modify the WPS JS adapter/PoC.

## Risks / rollback

The main risks are payload completeness, local experimental-data retention, workbook identity after Save As, and real Excel/tkinter behavior. The feature can be reverted as one PR; `~/.xstars/artifacts/` is derived cache data and may be deleted without changing workbook source data or `~/.xstars/settings.json`.
```

### 9.6 落盘前自查

- [x] 9 段结构齐全且顺序固定：Goal / Requirements / Non-goals / Research summary / Gap analysis / Milestone 表格 / To-do / Validation contract / 文件范围与治理。
- [x] Milestone 数量为 5，≤7 且 ≤10；全部初始 Status 为 `[ ]`。
- [x] G1–G10 均映射至少一个 To-do：**无孤立缺口**。
- [x] 每个 T1.1–T5.3 都包含文件、修改、验收、依赖。
- [x] Validation contract 包含命令、预期结果、通过标准、责任人和不可自动验证原因。
- [x] Git 策略包含 `feat/macos-support`、Draft PR 标题、可直接使用的描述、单 PR 决策、与 WPS PR 的独立关系及合并顺序。
- [x] 所有未获明确批准的格式、身份、保留、图表集合、输入语法、静态工具、实测矩阵与策略文档同步问题均进入待决事项，未擅自拍板。
- [x] 计划明确 Windows 零行为改动及所有现有 `ribbon/*.bas` 零修改。
- [x] 本次只新建 `plans/20260829-macos-support.md`，没有创建或覆盖源码、测试、配置、脚本或生成物。
- [x] 2026-08-30 批准修订：用户会话确认 Plan rev 批准，D1–D7 按推荐值定案；D8 改为“复制策略文档到本 PR（带前言）”，§9.1 文件范围相应新增一行；首个提交包含 Plan 与 Explore 调研报告。
- [x] 2026-08-30 批准范围扩展：用户批准在本 PR 内修复 `xstars/tools/standard_curve.py` 预先存在的 `back_calculate` NaN 掩码缺陷（main 基线确定性失败、相对 main 零 diff，与本 PR 无关但阻断完整回归门禁），§9.1 新增该文件行；修复仅限该缺陷。
