# XSTARS Windows WPS 支持评估与实施计划

> ⚠️ **SUPERSEDED（2026-08-30）**：本计划已被 [`plans/20260830-wps-support.md`](../plans/20260830-wps-support.md)（9 段式重梳 rev 1，已批准）取代。本文保留为 M0.0–M0.3 的历史证据与原始评估记录，不再作为实施契约。

> 状态：**已获用户批准（M0.1–M0.3 已通过；M0.3 关键发现：ShellExecute 失效，服务拉起改由安装器自启动）**
> 基准：`main` @ `5f4c409`，XSTARS `v1.1.1`
> 目标宿主：Windows 10/11 x64 + 最新稳定版 WPS 365/12.x 表格
> 阻断验收：WPS 专业版/商业版/政企版 x64
> Beta：WPS 个人版（尽力支持，不阻断首版发布）
> 约束：实施仅在 `feature/wps-support` 分支进行；M0 Gate 0 通过前不进行大规模核心重构。

## 1. Goal

在不改变现有 Microsoft Excel 功能和安装体验的前提下，为 XSTARS 新增一个独立的 Windows WPS 发行形态：

- WPS 端采用官方 JS 加载项（Ribbon + JavaScript）；
- 本地 Python 组件仅监听 `127.0.0.1`，完全离线运行；
- 复用现有统计、预设、绘图、Tkinter 配置和设置持久化能力；
- 在 WPS 中完成选区读取、分析、结果表写回、图片插入及高分辨率导出；
- 使用独立 `XSTARS_WPS_Setup.exe` 发布，现有 Excel 安装包保持不变。

## 2. Requirements

### 2.1 产品范围

1. Windows 10/11 x64。
2. 首版阻断支持最新稳定版 WPS 365/12.x 专业版、商业版或政企版。
3. WPS 个人版作为 Beta/尽力支持；不因个人版专有问题阻断专业版首发。
4. 保持现有 Excel 入口、Ribbon/VBA 回调、命令行调用及安装体验不回归。
5. WPS 使用独立安装包，不与 Excel 安装器合并。
6. 必须完全离线可用；允许仅本机回环 HTTP；允许安装 WPS JS 加载项和 `publish.xml`。
7. WPS 保留 XSTARS Ribbon；参数配置继续复用本地 Tkinter/ttkbootstrap 对话框。

### 2.2 WPS MVP 功能

- 常规 Run 与 Quick Run；
- WB、qPCR、CCK-8、ELISA；
- Transform Only 与 Standard Curve；
- 主题、配色和 `~/.xstars/settings.json` 持久化；
- 统计表、转换数据和图片写回工作表；
- 选中图片的高分辨率导出：PNG、TIFF、JPG、PDF，自定义 DPI；
- 用户友好错误提示、结构化日志和诊断能力；
- `XSTARS_Templates.xlsx` 的全部 Sheet/示例流程可在 WPS 中执行。

### 2.3 分支与 PR 交付工作流

1. 每个新功能在开始实施前，必须从最新 `main` 创建独立 `feature/<short-name>` 分支。
2. 分支首次形成可审阅骨架后立即推送，并创建 **Draft PR**；不得在 `main` 上直接开发功能。
3. 所有 milestone 的实现、验证证据、已知风险和 checklist 状态持续更新到同一个 Draft PR。
4. 只有批准范围内的功能完成、必要 CI/实机验证通过、Review 无 Blocker 或“现在值得修复”事项、父 Agent 检查最终 diff 后，Draft PR 才可转为 Ready。
5. Ready PR 必须在要求的检查全部通过后，以 **squash merge** 合并到 `main`，并删除功能分支。
6. 未通过验收、仍有待决产品问题或只完成 PoC 时，不得合并到 `main`。

## 3. Non-goals

首版明确不包含：

- macOS WPS；
- WPS WebOffice、金山文档网页端或移动端；
- 32 位 WPS/Python；
- 重写现有统计方法、预设算法或绘图风格；
- 将 Tkinter 配置 UI 全面重写为 WPS 任务窗格；
- 将 Excel 端迁移到 WPS JS/HTTP 架构；
- 单一安装包自动安装 Excel 与 WPS 两套组件；
- 将非官方 VBA 补丁作为个人版依赖；
- 云服务、账号系统、遥测或外部网络调用。

## 4. Current-state evidence

### 4.1 当前调用链

```text
Excel Ribbon/VBA
  -> RunPython 或 Shell xstars.exe <command> <workbook_path>
  -> xstars/cli.py
  -> xlwings.Book(...).set_mock_caller()
  -> xstars/main.py
  -> DataHandler / Presets / StatsEngine / PlotEngine
  -> xlwings Range/Pictures/StatusBar 写回 Excel
```

关键耦合点：

- `xstars/cli.py:16-36`：通过 `xw.Book(workbook_path)` 绑定 Excel；
- `xstars/data_handler.py:26-90`：直接使用 `Book.caller()`、Selection、Range；
- `xstars/main.py:205-1507`：入口、写回、状态栏、图片插入、InputBox、Shape 导出均依赖 xlwings/Excel COM；
- `ribbon/customUI14.xml`、`ribbon/*.bas`：Excel Ribbon/VBA；
- `ribbon/ribbon_callbacks_installed.bas`：从注册表读取安装目录并 Shell 启动 exe。

可直接复用的核心：

- `xstars/stats_engine.py`；
- `xstars/plot_engine.py`、`xstars/styles.py`、`xstars/annotations.py`；
- `xstars/presets/*`、`xstars/tools/*`；
- `xstars/config.py`；
- 大部分 Tkinter 对话框。

### 4.2 测试基线

计划制定时执行：

```text
python -m pytest -q
EXIT 0
142 passed, 3 warnings in 4.68s
```

现有 `tests/test_end_to_end.py` 使用 mocked xlwings 验证 Quick Run 编排和图片插入，但仓库没有真实 WPS 集成测试。

`XSTARS_Templates.xlsx` 当前包含：

1. `Data`
2. `WB`
3. `qPCR`
4. `CCK8`
5. `ELISA`
6. `Transform`
7. `Standard Curve`

## 5. Research summary

### 5.1 已确认事实

1. **WPS JS 加载项结构**：官方文档定义 `ribbon.xml` + 自动生成的 `index.html` + `main.js`。
2. **选区和单元格值**：ET JS API 提供 `Application.Selection`；单元格选区返回 Range；`Range.Value2` 可读写。
3. **本地图片插入**：ET `Shapes.AddPicture(Filename, LinkToFile, SaveWithDocument, Left, Top, Width, Height)` 支持从本地文件创建并嵌入图片。
4. **本地进程拉起**：`OAAssist.ShellExecute(Url, Params)` 官方说明可打开网页或调起进程；被调起进程不会随当前进程自动终止。
5. **WPS 加载项部署**：官方推荐 `publish.xml` 模式；从 `12.1.0.16910` 起旧 `jsplugins.xml` 模式受限。
6. **官方 publish 模式版本说明只明确列出企业版分支**。因此个人版 JS 加载项部署能力不能仅凭文档保证，必须实机验证；个人版只作为 Beta 是合理边界。
7. **xlwings 不正式支持 WPS**，不应以修改 `Excel.Application` ProgID 的 monkey patch 作为产品主路线。

### 5.2 尚未由官方资料完整证明、必须 PoC 的事项

- WPS 加载项页面的实际 Origin、原生 `fetch` 到 `127.0.0.1` 的 CORS 行为；
- 专业版及个人版中 `OAAssist.ShellExecute` 的权限和静默程度；
- 官方工具生成的离线 `publish.xml`/压缩包结构，不能在实现前假定手写 XML 格式；
- WPS 个人版是否允许相同方式安装并长期启用加载项；
- Shape 选中、`ShapeRange`、`CopyPicture`/Export 与 Windows 剪贴板的行为；
- 任意选中图片按自定义 DPI 导出到 PNG/TIFF/JPG/PDF 的精确等价性；
- ELISA 二次范围选择的最佳 WPS 交互；
- 长时间等待 Tkinter 对话框时，WPS JS Promise/HTTP 请求是否保持稳定；
- WPS 多屏/高 DPI 下 Tkinter 前置和图片定位。

### 5.3 主要参考资料

- WPS 加载项开发说明：<https://open.wps.cn/documents/app-integration-dev/wps365/client/wpsoffice/wps-integration-mode/wps-addin-development/wps-addin-development-instructions>
- WPS `Application.Selection`：<https://open.wps.cn/documents/app-integration-dev/wps365/client/wpsoffice/jsapi/et/Application/member/Selection>
- WPS `Range.Value2`：<https://open.wps.cn/documents/app-integration-dev/wps365/client/wpsoffice/jsapi/et/Range/member/Value2>
- WPS `Shapes.AddPicture`：<https://open.wps.cn/documents/app-integration-dev/wps365/client/wpsoffice/jsapi/et/Shapes/member/AddPicture>
- WPS `OAAssist.ShellExecute`：<https://open.wps.cn/documents/app-integration-dev/wps365/client/wpsoffice/jsapi/addin-api/OAAssist/member/ShellExecute>
- xlwings 文档：<https://docs.xlwings.org/en/stable/>

## 6. Options and recommendation

### 6.1 方案对比

| 方案 | 覆盖 | 改造量 | 主要问题 | 结论 |
| --- | --- | ---: | --- | --- |
| VBA + WPS COM/xlwings patch | 主要是带 VBA 的专业/政企版 | 中 | 个人版不可依赖；xlwings 高级行为不受支持 | 不作为主路线；可作为高清导出或诊断的受控 fallback |
| WPS JS 加载项 + 本地 Python 服务 | 专业版为正式目标，个人版可 Beta 验证 | 中高 | 部署、CORS、生命周期、Shape 导出需 PoC | **推荐** |
| 独立桌面应用导入/导出 xlsx | 不依赖加载项 | 高 | 失去“一键、原位写回”的核心体验 | 仅作为个人版加载项不可用时的后备产品方向 |

### 6.2 推荐目标架构

```text
WPS ET
  └─ wps-addon/
      ├─ Ribbon 回调
      ├─ 读取 Selection.Value2/Address
      ├─ 调起或探测本地服务
      ├─ POST JSON 命令
      └─ 按 WritebackPlan 写单元格并 AddPicture

127.0.0.1 本地 broker
  ├─ 仅回环绑定
  ├─ 每安装实例随机密钥
  ├─ Origin/CORS 白名单（以 PoC 观测值为准）
  ├─ 单任务互斥与端口发现
  ├─ 启动分析 worker
  └─ 返回结构化结果/错误/临时图片路径

分析 worker（独立子进程）
  ├─ 在主线程显示 Tkinter 对话框
  ├─ 调用共享 application/use-case 层
  ├─ 生成统计表、转换数据、图像和导出文件
  └─ 通过结果文件/IPC 返回 broker

共享 Python application 层
  ├─ 输入：SelectionPayload + command + settings
  ├─ 复用 DataHandler/Presets/StatsEngine/PlotEngine
  └─ 输出：WritebackPlan + artifacts + status

Excel 端
  └─ 保留现有 Ribbon/VBA/xlwings 入口；分阶段接入同一 application 层
```

选择 broker + worker，而不是在 HTTP 处理线程直接创建 Tkinter 窗口，原因是 Tkinter 需要稳定的 GUI 主线程；独立 worker 也能隔离绘图、剪贴板或 UI 崩溃，并最大程度复用现有“每次命令启动一个 exe”的执行模式。

### 6.3 安全约束

- 服务只能绑定 `127.0.0.1`，测试必须证明未监听 `0.0.0.0`/局域网地址；
- 安装时生成每用户随机密钥，同时写入服务配置与加载项生成配置，防止普通网页 CSRF 调用；
- 不使用 `Access-Control-Allow-Origin: *` 作为发布配置；PoC 先记录真实 Origin，再建立最小白名单；
- 命令使用枚举，不允许客户端传任意 Python 函数名、shell 命令或输出路径穿越；
- 所有临时文件使用受控目录，按任务清理；日志默认不记录完整实验数据；
- 完全离线，不包含外部 API、遥测或更新检查。

## 7. Gap analysis

| 能力 | 当前状态 | WPS 缺口 | 推荐处理 | 风险 |
| --- | --- | --- | --- | --- |
| Run/Quick Run | Excel/xlwings 完成 | 无 WPS Selection/写回入口 | 共享 use-case + WPS DTO/WritebackPlan | 中 |
| WB/qPCR/CCK-8 | 核心算法可复用 | 宿主数据/图片桥接 | 复用算法，新增契约测试 | 低至中 |
| ELISA | Excel `InputBox(Type=8)` 二次选区 | WPS 交互未定义 | PoC 比较 WPS InputBox、两阶段 Ribbon 流程、地址输入 | 高 |
| Transform Only | 纯变换 + Excel 写回 | WPS 批量 Value2 写回 | WritebackPlan | 低 |
| Standard Curve | 核心可复用 | 图表与结果写回 | WritebackPlan + AddPicture | 中 |
| 设置/主题 | JSON 与宿主无关 | WPS Ribbon 回调缺失 | WPS 菜单调用受限命令 | 低 |
| 图片插入 | xlwings Pictures | WPS Shapes API | 本地 PNG + AddPicture | 中，需实机定位验证 |
| 高清导出 | Excel Shape 放大 + CopyPicture + ImageGrab | WPS 任意 Shape 导出未证明 | Gate 0 先验证；必要时专业版使用 ET COM/剪贴板，XSTARS 图使用原始渲染缓存 | **高/可能阻断** |
| 错误/状态 | Tkinter + Excel StatusBar | JS 与服务错误映射缺失 | 稳定错误码、WPS alert/status、日志 | 中 |
| 服务生命周期 | 无 daemon | 拉起、端口、单实例、退出 | broker + authenticated health + idle exit | 中 |
| 离线部署 | 仅成品 Excel 安装器，源码未跟踪 | WPS 发布/卸载流程缺失 | 官方 wpsjs 构建 + 独立安装器 | 高 |
| 个人版 | 无支持 | 官方 publish 版本说明未明确覆盖 | Beta 实机验证；失败不阻断专业版 | 高但非阻断 |
| 自动测试 | 142 个 Python 测试 | 无 JS/HTTP/WPS 实机层 | 新增契约、服务、JS mock、真实宿主验收 | 中 |

## 8. Validation contract

只有满足相应里程碑合同，任务才允许从 `[ ]` 更新为 `[x]`。

### 8.1 自动化合同

1. `python -m pytest -q`：现有 142 个测试全部通过；允许现有 3 个已知 warning，但不得新增未解释 warning。
2. 新增 application/DTO/服务测试全部通过，包括：
   - 单单元格与二维 Selection 归一化；
   - 非连续选区拒绝；
   - 数据类型、NaN/null、Unicode 组名；
   - 所有命令枚举及错误码；
   - Token 缺失/错误、CORS 预检、端口占用、重复服务、并发任务；
   - 路径穿越、非法输出路径、超大请求和临时文件清理；
   - worker 取消、UI 取消、超时和崩溃恢复。
3. JS 单元测试使用 mock WPS API 验证：Selection 序列化、WritebackPlan 坐标、Value2 写入、AddPicture 参数、错误映射和服务发现。
4. Excel characterization/mock 测试保持通过；真实 Excel 至少完成现有模板 smoke test，证明现有安装/入口未变化。

### 8.2 PoC Gate 0（正式重构前的硬门槛）

在真实 WPS 专业/商业版 x64、完全断网环境中确认：

- [ ] 官方方式安装/启用最小加载项，Ribbon 按钮显示并回调；
- [ ] `Selection`、`Address`、`Value2` 二维数据读写；
- [ ] 观测插件 Origin，`fetch` 回环服务可完成预检和 JSON 往返；
- [ ] `OAAssist.ShellExecute` 能启动签名/未签名测试 exe，失败时错误可诊断；
- [ ] `Shapes.AddPicture` 可嵌入本地 PNG，保存/重开工作簿后仍存在；
- [ ] Tkinter 打开期间 WPS 不冻结，请求可取消/恢复；
- [ ] ELISA 二次范围选择至少有一种可接受交互；
- [ ] 选中图片可被识别，并评估四格式/自定义 DPI 导出路径；
- [ ] 个人版执行相同 smoke test并记录能力矩阵（非阻断）。

若专业版中加载项离线部署、回环调用或图片插入任一失败，暂停后续实施并重新决策架构。若任意图片高清导出无法等价实现，必须回到用户处决定是否接受“仅 XSTARS 生成图可重渲染导出”或专业版 COM fallback，不能静默缩小需求。

### 8.3 真实 WPS 阻断验收

在 Windows 10 x64 和 Windows 11 x64、最新稳定 WPS 专业/商业/政企版中：

- 独立安装包安装后无需网络即可显示 XSTARS Ribbon；
- `XSTARS_Templates.xlsx` 的 `Data`、`WB`、`qPCR`、`CCK8`、`ELISA`、`Transform`、`Standard Curve` 全部流程通过；
- 数值结果与 Excel/Python 基准在算法容差内一致；统计方法、比较对和显著性结论一致；
- 图片嵌入、命名、位置不覆盖源数据/结果表，保存重开后仍存在；
- PNG/TIFF/JPG/PDF 在所选 DPI 下生成，像素尺寸和 DPI 元数据经独立读取验证；
- 无效数据、取消对话框、服务未启动、端口冲突时 WPS 不崩溃且给出可行动错误；
- 断网运行无外部请求和等待；
- 卸载后移除本产品的加载项/服务文件，但不破坏用户其他 `publish.xml` 项或设置；
- Excel 现有模板 smoke test与 142+ 自动化测试仍通过。

个人版执行同一矩阵并形成 Beta 能力说明；失败项记录在发布说明，不阻断专业版发布。

## 9. File-level modification scope

以下是计划范围，Gate 0 后可根据官方脚手架的实际文件名微调，但不得扩大产品范围。

### 9.1 预计新增

| 路径 | 用途 |
| --- | --- |
| `poc/wps/` | Gate 0 最小加载项、回环服务探针、Shape/导出探针；通过后保留为验证资产或迁移到正式目录 |
| `xstars/application/contracts.py` | `SelectionPayload`、命令枚举、`WritebackPlan`、错误/Artifact DTO |
| `xstars/application/analysis.py` | 宿主无关的 Run/Quick/预设/Transform/Standard Curve 用例 |
| `xstars/application/export.py` | 导出命令、格式/DPI校验、原始渲染与 WPS fallback 协调 |
| `xstars/application/worker.py` | 从受控请求文件执行单个命令；在主线程运行 Tkinter；原子写结果 |
| `xstars/wps_service.py` | 127.0.0.1 broker、鉴权、CORS、端口、单实例、worker 管理、健康检查 |
| `wps-addon/ribbon.xml` | WPS Ribbon 定义 |
| `wps-addon/main.js` | 加载项启动和 Ribbon 回调 |
| `wps-addon/service-client.js` | 服务发现、鉴权、请求/取消、错误映射 |
| `wps-addon/spreadsheet.js` | Selection 序列化、Value2 写回、Shapes.AddPicture、状态提示 |
| `wps-addon/config.template.js` | 安装器注入端口范围与每安装实例随机密钥的模板 |
| `wps-addon/assets/` | 自有图标，避免依赖 `imageMso` |
| `wps-addon/package.json` | wpsjs 构建及 Node mock 测试命令 |
| `wps-addon/tests/` | WPS API mock、序列化、写回和 RPC 客户端测试 |
| `installer/wps/xstars-wps.spec` | WPS service/worker PyInstaller 规格 |
| `installer/wps/XSTARS_WPS.iss` | 独立安装、升级、卸载与配置合并 |
| `installer/wps/build.ps1` | 可重复构建脚本 |
| `tests/test_application_contracts.py` | DTO和命令边界测试 |
| `tests/test_application_analysis.py` | 宿主无关用例测试 |
| `tests/test_wps_service.py` | HTTP、安全、生命周期和 worker 测试 |
| `tests/test_wps_export.py` | 格式、DPI、路径和 fallback 测试 |
| `docs/wps-installation.md` | WPS 安装、版本、Beta限制与卸载 |
| `docs/wps-validation.md` | 实机矩阵、模板逐项结果和证据模板 |

### 9.2 预计修改

| 路径 | 修改 |
| --- | --- |
| `xstars/main.py` | 分阶段将业务计算委托给 application 层；保留所有现有公开入口和 Excel 写回行为 |
| `xstars/data_handler.py` | 新增从二维值/DTO构造 DataFrame 的入口；保留现有 xlwings 方法 |
| `xstars/cli.py` | 增加受限 `serve`/`worker` 模式；旧 `<command> <workbook_path>` 语法保持不变 |
| `xstars/config.py` | 如需增加 WPS service 配置，只使用向后兼容字段；统计/主题设置格式不破坏 |
| `pyproject.toml` | 增加打包入口和必要依赖；优先标准库服务，避免无必要 Web 框架依赖 |
| `tests/test_end_to_end.py` | 增加 Excel characterization，确保 application 抽取不改变当前调用 |
| `README.md`、`README.zh-CN.md` | 实施完成且验收后再记录 WPS 支持范围、专业版正式/个人版 Beta |

### 9.3 原则上不修改

- `ribbon/customUI14.xml`、`ribbon/*.bas`；
- 现有 Excel 安装包/制品；
- `xstars/stats_engine.py`、`plot_engine.py`、`styles.py`、`annotations.py`；
- `xstars/presets/*`、`xstars/tools/*` 的算法，除非测试证明现有宿主耦合且变更不改变算法结果。

## 10. Milestones

| Milestone | Status | Dependencies | Validation | Notes |
| --- | --- | --- | --- | --- |
| M0 — WPS feasibility PoC | Not started | 用户批准、`feature/wps-support` 分支、Draft PR、真实专业版和个人版测试机 | Gate 0 全部专业版硬门槛 | 不进行大规模核心重构；失败立即停线 |
| M1 — Characterization + application contracts | Not started | M0 通过 | 142 基线 + 新契约/Excel回归测试 | 先锁定行为再抽取 |
| M2 — Local broker + worker | Not started | M1 | HTTP、安全、生命周期、UI取消测试 | Tkinter 在 worker 主线程 |
| M3 — WPS Ribbon + Run/Quick writeback | Not started | M2 | 真实 WPS Data 场景 + 保存重开 | 形成首个垂直切片 |
| M4 — Presets, ELISA, tools, themes, export | Not started | M3 | 全功能自动化 + 专项实机 | 高清导出、ELISA为高风险 |
| M5 — Installer, diagnostics, offline hardening | Not started | M4 | 干净 Win10/11 VM 安装/升级/卸载 | 不覆盖其他加载项配置 |
| M6 — Full template acceptance + Beta release | Not started | M5 | 专业版阻断矩阵全绿；个人版能力报告 | Review完成后才能发布 |

## 11. Markdown task checklist

- [x] **M0.0 创建功能分支并开启 Draft PR**
  - 文件：Git 分支、Draft PR 描述、`docs/wps-support-implementation-plan.md`
  - 修改：从最新 `main` 创建 `feature/wps-support`，推送后立即创建 Draft PR；在 PR 中引用本计划和验证合同。
  - 验收：开发不发生在 `main`；Draft PR 可见，初始 checklist、范围、风险和验证计划完整；证据：Draft PR [#1](https://github.com/Frankkk1912/xstars/pull/1)。
  - 依赖：用户明确批准本计划。

- [x] **M0.1 建立最小 WPS JS 加载项与官方离线部署 PoC**
  - 文件：`poc/wps/addin/*`、`poc/wps/README.md`
  - 修改：使用官方 wpsjs 工具生成 Ribbon；记录实际生成的 publish 配置和插件 Origin，不手写未经验证的格式。
  - 验收：专业版断网可安装、显示 Ribbon、执行按钮；个人版结果记录为 Beta 能力。
  - 证据：WPS 365 教育高级版 `12.1.0.28022` 64 位在完全断网环境通过官方 `publish` 安装、Ribbon 显示、按钮回调、完全退出后重启回调和卸载；实际 Origin 为 `file://`；记录见 Draft PR [#1](https://github.com/Frankkk1912/xstars/pull/1#issuecomment-5461289789)。
  - 依赖：M0.0；专业版和个人版测试机。

- [x] **M0.2 验证 Selection/Value2/AddPicture 垂直链路**
  - 文件：`poc/wps/addin/*`、`poc/wps/probe_server.py`
  - 修改：选区二维值发往回环探针，返回矩阵和 PNG 路径，写回并嵌入图片。
  - 验收：保存并重开后数据和图片仍存在；记录坐标、缩放和错误行为。
  - 证据：2026-08-29 完全断网实机验证，WPS 365 教育高级版 `12.1.0.28022` 64 位：官方 `publish` 安装 `1.1.0` 后，`A1:B2` 二维 `Value2` 经 Origin `null` 预检发往 `127.0.0.1:3891/probe`，JSON 往返成功；`D1:E2` 写回矩阵一致；`Shapes.AddPicture` 嵌入 320×180 PNG（11.29 × 6.35 厘米，`D1:E2` 下方）；另存副本、完全退出重开后值和图片均存在；全程无报错。
  - 依赖：M0.1。

- [x] **M0.3 验证服务拉起、CORS、Tkinter 与生命周期**
  - 文件：`poc/wps/*`
  - 修改：测试 `OAAssist.ShellExecute`、真实 Origin、预检、端口冲突、Tkinter阻塞/取消。
  - 验收：服务未运行时可恢复；WPS 不冻结；失败有可诊断结果。
  - 证据：2026-08-29 实机验证（宿主同前）。**关键负面发现：`OAAssist.ShellExecute` 在本宿主弹安全确认窗后静默不启动目标进程**（入口日志 + 143 次采样无痕迹；官方 2 参数签名、修正后依旧失效；与社区接口审查下线报告一致）→ 服务拉起策略改为 M5.1 安装器自启动，加载项仅健康检查与引导。其余全过：真实 Origin `file://` 预检与 JSON 往返成功；Tkinter 对话框期间 WPS 不冻结且取消正确回报（`confirmed=False`）；端口冲突可诊断（`SO_EXCLUSIVEADDRUSE` 单实例，双开产生 `PORT CONFLICT` 日志并以退出码 2 退出，原服务不受影响）；服务被杀后外部可恢复。M2.1 设计输入：服务自启动由安装器负责。
  - 依赖：M0.2。

- [ ] **M0.4 验证 ELISA 二次选区与高清导出可行性**
  - 文件：`poc/wps/elisa_selection.*`、`poc/wps/shape_export.*`
  - 修改：比较 WPS InputBox、两阶段交互和地址输入；验证选中 Shape、CopyPicture/Export/COM fallback。
  - 验收：ELISA 有可接受流程；PNG/TIFF/JPG/PDF 自定义 DPI 有等价路径，或暂停并向用户请求缩小范围。
  - 依赖：M0.3。

- [ ] **M1.1 增加 Excel 行为刻画测试**
  - 文件：`tests/test_end_to_end.py`、必要的新 fixture
  - 修改：覆盖现有入口、选区解析、结果表位置、图片命名、错误、导出与取消行为。
  - 验收：重构前测试可稳定复现当前行为；142 个基线测试通过。
  - 依赖：M0 全部通过。

- [ ] **M1.2 定义宿主无关请求/响应契约**
  - 文件：`xstars/application/contracts.py`、`tests/test_application_contracts.py`
  - 修改：定义命令白名单、SelectionPayload、WritebackPlan、Artifacts和稳定错误码；提供版本字段。
  - 验收：序列化往返、非法输入、路径和大小边界测试通过。
  - 依赖：M1.1。

- [ ] **M1.3 抽取共享 application 用例并保持 Excel 回归**
  - 文件：`xstars/application/analysis.py`、`xstars/main.py`、`xstars/data_handler.py`、`tests/test_application_analysis.py`
  - 修改：将计算/产物生成与 xlwings 写回分离；Excel 入口和函数签名保持不变。
  - 验收：142+测试全绿；真实 Excel 模板 smoke test无行为回归。
  - 依赖：M1.2。

- [ ] **M2.1 实现本地 broker 安全边界**
  - 文件：`xstars/wps_service.py`、`tests/test_wps_service.py`
  - 修改：仅回环监听、每安装实例密钥、Origin白名单、命令白名单、请求限制、健康检查、单实例和端口范围。
  - 验收：安全/错误/端口/并发测试通过；证明未暴露到局域网。
  - 依赖：M1.3、M0 Origin证据。

- [ ] **M2.2 实现单任务 worker 与 GUI 主线程模型**
  - 文件：`xstars/application/worker.py`、`xstars/cli.py`、相关测试
  - 修改：broker 启动受控 worker；原子请求/结果文件；取消、超时、崩溃和临时文件清理。
  - 验收：Run/Quick mock E2E；Tkinter取消无残留；旧 CLI 语法测试通过。
  - 依赖：M2.1。

- [ ] **M3.1 实现 WPS Ribbon 和服务客户端**
  - 文件：`wps-addon/ribbon.xml`、`main.js`、`service-client.js`、`config.template.js`、`assets/*`
  - 修改：复刻批准范围内 Ribbon；服务探测/拉起/鉴权/错误展示；不依赖 `imageMso`。
  - 验收：JS mock测试通过；专业版真实 Ribbon 和 Run/Quick 可用。
  - 依赖：M2.2。

- [ ] **M3.2 实现 WPS Selection 和 WritebackPlan 执行器**
  - 文件：`wps-addon/spreadsheet.js`、`wps-addon/tests/*`
  - 修改：连续选区验证、Value2序列化、批量写回、图片命名/定位、状态栏与错误映射。
  - 验收：Data Sheet Run/Quick 完整垂直切片；保存重开结果不丢失。
  - 依赖：M3.1。

- [ ] **M4.1 接通 WB/qPCR/CCK-8、Transform 和 Standard Curve**
  - 文件：`xstars/application/analysis.py`、`wps-addon/main.js`、相关测试
  - 修改：增加各命令映射和特定 WritebackPlan；核心算法保持不变。
  - 验收：模板对应 Sheet 在专业版逐项通过，数值与 Python/Excel基准一致。
  - 依赖：M3.2。

- [ ] **M4.2 完成 ELISA 两阶段交互**
  - 文件：`xstars/application/analysis.py`、`wps-addon/main.js`、`spreadsheet.js`、相关测试
  - 修改：实现 M0 选定的标准数据/样本数据选择协议和取消恢复。
  - 验收：ELISA 模板完整拟合、反算、统计、图表和可选标准曲线通过。
  - 依赖：M4.1、M0.4结论。

- [ ] **M4.3 完成主题/设置与错误诊断**
  - 文件：`wps-addon/ribbon.xml`、`main.js`、`xstars/config.py`、`xstars/wps_service.py`
  - 修改：主题命令、设置共享、稳定错误码、脱敏日志和诊断包。
  - 验收：设置跨 WPS 重启持久化；故障用例不崩溃且日志可定位。
  - 依赖：M4.1。

- [ ] **M4.4 完成高分辨率导出**
  - 文件：`xstars/application/export.py`、`tests/test_wps_export.py`、`wps-addon/main.js`
  - 修改：实现 M0 验证的 Shape识别/COM或原始渲染路径；严格校验格式、DPI和目标路径。
  - 验收：选中图片导出 PNG/TIFF/JPG/PDF；独立读取验证 DPI、像素和文件有效性。
  - 依赖：M0.4、M3.2。

- [ ] **M5.1 构建 WPS 独立可重复安装包**
  - 文件：`installer/wps/xstars-wps.spec`、`XSTARS_WPS.iss`、`build.ps1`
  - 修改：打包 service/worker/add-in；生成每安装实例密钥；按官方机制安装/升级/卸载加载项；用户级目录优先。
  - 验收：干净 Win10/11 x64 可安装、升级、卸载；不影响 Excel 或其他 WPS 加载项。
  - 依赖：M4 全部完成。

- [ ] **M5.2 完善用户文档和版本声明**
  - 文件：`docs/wps-installation.md`、`README.md`、`README.zh-CN.md`
  - 修改：记录专业版正式支持、个人版 Beta、版本基线、离线安装、故障诊断和卸载。
  - 验收：文档步骤在干净测试机可复现；不夸大个人版支持。
  - 依赖：M5.1。

- [ ] **M6.1 执行专业版阻断矩阵与个人版 Beta 矩阵**
  - 文件：`docs/wps-validation.md`
  - 修改：逐 Sheet、逐功能、逐系统记录版本、命令、截图/文件哈希和结果。
  - 验收：专业版全部阻断项通过；个人版差异被明确记录。
  - 依赖：M5.2。

- [ ] **M6.2 完成 fresh-context Review、修复与最终 diff 检查**
  - 文件：所有变更文件、Draft PR
  - 修改：并行审查正确性/回归、测试、可维护性，并增加安全、用户流程和安装/文档契约审查；仅由单一 fix worker 处理“现在值得修复”。
  - 验收：无 Blocker 或立即值得修复项；最多三轮；父 Agent 检查最终 diff；Draft PR 满足转 Ready 条件。
  - 依赖：M6.1。

- [ ] **M6.3 将通过验收的 PR 合并到 main**
  - 文件：Draft PR、`main` 分支
  - 修改：将 Draft PR 转为 Ready；确认所有要求检查通过后 squash merge，并删除 `feature/wps-support` 分支。
  - 验收：合并后的 `main` CI/必要 smoke test 通过；PR 保留完整验证证据；没有未经批准的剩余任务被错误标记完成。
  - 依赖：M6.2。

## 12. Risks and mitigations

| 风险 | 等级 | 缓解/决策门槛 |
| --- | ---: | --- |
| 官方 publish 版本说明仅明确企业版，个人版可能不能安装加载项 | 高 | 个人版明确为 Beta；M0 实机形成版本能力矩阵，不夸大支持 |
| 任意选中图片的四格式/自定义 DPI 等价导出不可行 | 高/阻断 | M0.4 前置；优先验证 WPS Shape/剪贴板和专业版 ET COM；失败必须向用户重新确认范围 |
| 手写 `publish.xml` 与官方当前格式不一致 | 高 | 只以当前官方 wpsjs 生成物和真实安装流程为准 |
| Tkinter 从 HTTP 线程调用不安全或焦点丢失 | 高 | broker + 独立 worker；UI 在 worker 主线程；M0 做多屏/取消测试 |
| WPS JS Origin/CORS 与 Chromium 假设不同 | 中高 | M0 实测 Origin；发布时最小白名单，不使用通配符 |
| `ShellExecute` 被个人版或 EDR 阻止 | 中高 | 专业版阻断测试；提供手动启动/协议唤醒备选；签名纳入发布决策 |
| 多实例 WPS/并发点击造成错误写回 | 中 | 请求携带工作簿/Sheet/选区快照和 job id；单任务锁；写回前检查上下文 |
| WPS API/版本更新导致回归 | 中 | 固定“开发时最新稳定 12.x”基线，记录精确 build；每次发布重跑实机矩阵 |
| 临时图和日志泄露实验数据 | 中 | 受控目录、最短保留、脱敏日志、退出/启动清理 |
| 当前 Excel 安装器源码未在仓库中，难以做统一构建验证 | 中 | 首版 WPS 安装器完全独立；不修改现有 Excel 制品；文档记录限制 |
| 未签名 PyInstaller/Inno Setup 被 EDR 误报 | 中 | 干净机/政企环境预检；发布前决定代码签名；提供校验和 |

## 13. Rollback plan

1. **M0 失败**：删除/停用 PoC 加载项，不修改 Excel 或核心；重新选择 COM-only 或桌面应用方向。
2. **M1 回归**：application 抽取保持小提交/小 diff；Excel 入口保留旧路径，未通过 characterization 前不切换。
3. **M2/M3 失败**：停用 WPS service/add-in；共享核心和 Excel 路径仍可独立发布。
4. **M4 单功能失败**：不得把未完成阻断功能标记完成；高清导出或 ELISA 若无法满足，暂停并请求用户决策，不以隐藏降级替代。
5. **安装器失败**：保留开发/便携验证流程；不发布 Setup，不触碰现有 Excel 安装包。
6. **发布回滚**：WPS 独立版本可单独卸载/回退；安装器必须备份并恢复其修改的加载项配置，不删除其他产品条目。

## 14. Open technical decisions

这些是实施阶段通过 M0/M1 证据解决的技术问题，不需要当前继续猜测产品决策：

1. 专业版和个人版的精确 WPS build 号及 JS 加载项能力差异；
2. 官方离线 publish 包的真实目录/配置格式；
3. 插件 Origin 与最小 CORS 白名单；
4. 服务固定端口、端口范围扫描或官方配置注入的最终机制；
5. `ShellExecute`、自定义 URL 协议、手动启动三者的优先级；
6. ELISA 二次范围选择采用 WPS InputBox、两阶段 Ribbon，还是地址输入；
7. 任意 Shape 高清导出采用 JS、ET COM/剪贴板，还是可证明等价的组合方案；
8. 安装包是否必须代码签名（若目标 EDR 环境要求，将成为发布前阻断项）。

## 15. Approval gate

本文件同时作为评估记录、已批准的实施范围和验证合同。

**用户已明确批准本计划；实施从最新 `main` 的 `feature/wps-support` 分支开始，并通过 Draft PR 持续记录范围、风险和验证证据。** 实施按 M0 → M6 串行推进；每个 milestone 结束必须先验证并报告，再继续下一阶段。全部批准范围完成并通过验证、Review 与最终 diff 检查后，才将 PR 转为 Ready 并 squash merge 到 `main`。
