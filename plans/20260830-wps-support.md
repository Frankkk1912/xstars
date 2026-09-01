# XSTARS Windows WPS 支持实施 Plan（9 段式重梳）

- **状态**：已批准（2026-08-30 用户会话明确批准 rev 1）
- **计划日期**：2026-08-30
- **性质**：对已批准旧计划 `docs/wps-support-implementation-plan.md` 的**全范围重梳与接续**——继承其已完成项（M0.0–M0.3）与实机证据，将 M0.4 + 旧 M1–M6 重组为 7 个 Milestone。本计划不是新功能。
- **基准**：`main` @ `5f4c409`，XSTARS `v1.1.1`
- **目标宿主**：Windows 10/11 x64 + 最新稳定版 WPS 365/12.x 表格；阻断验收 = WPS 专业版/商业版/政企版 x64；Beta = WPS 个人版
- **目标分支**：`feature/wps-support`（沿用，不新建）
- **目标 PR**：Draft PR [#1](https://github.com/Frankkk1912/xstars/pull/1)「feat: add Windows WPS support」（沿用，标题不变）

## Changelog

| rev | 日期 | 说明 |
| --- | --- | --- |
| rev 1 | 2026-08-30 | 由旧文档 `docs/wps-support-implementation-plan.md` 重梳而来：继承 M0.0–M0.3 已完成项与实机证据；记录 2026-08-30 访谈决策 D-A..D-H；将 M0.4+旧 M1–M6 重组为 7 个 Milestone；沿用 mac-support PR 的 9 段式风格与 feature-impl PR 模板。 |
| rev 1（批准） | 2026-08-30 | 用户对本精确 rev 明确回复「批准」；进入 `/feature-impl` 实施，从 M1（Gate 0 收尾）开始，每 Milestone 验证闸门通过后才进入下一个。 |
| rev 1（白名单扩展） | 2026-08-31 | 用户批准将 `poc/wps/addin/package.json` 纳入 T1.3 文件白名单，仅限 PoC 版本号随 Milestone 递增（本次 1.2.2→1.2.3）；背景：M0.4 实机步骤 4 发现 WPS 内嵌浏览器不支持 `window.prompt`（返回 null 被误判为取消，commit `97820ce` 已改用宿主原生 `InputBox(Type=2)`），且同版本重装会跳过解压，需版本递增触发重新提取。 |
| rev 1（白名单扩展二） | 2026-08-31 | M4 实施中发现 §9.1 未列 `wps-addon/` 的官方 wpsjs 壳文件（index.html/manifest.xml/vite.config.js/scripts/inject-config.cjs）；编排器据旧计划「官方脚手架实际文件名可微调」条款批准补入 §9.1 新建表（零业务逻辑壳，否则 T4.3 无法打包实机验证），待用户确认。 |
| rev 1（T5.4 范围定案） | 2026-08-31 | 用户选择方案 A：M5.4 高分辨率导出按 M0.4 O3 结论完整实现——worker 生成图表时 best-effort 持久化轻量重渲染 payload（~/.xstars/artifacts，由 worker 下发 pictureId、加载项重命名 Shape 关联），导出时 Python 重渲染（XSTARS 图主路径，真细节）；任意图片剪贴板重编码为 bonus。§7 T5.4 文本更新；`worker.py`/`analysis.py` 的 M5 修改与 `inject-config.cjs` 端口读取纳入范围。 |
| rev 1（范围修订） | 2026-08-31 | ① T5.5 实机矩阵全部通过（WB/qPCR 标签、ELISA、预设、主题设置、导出；过程中修复标签列检测缺失与 job 目录误删两缺陷）；② 用户决定：M6（独立安装器/文档/离线加固）**移出本 PR 范围**，延后到功能改进稳定后的独立 PR；③ M7 调整为「Excel↔WPS 功能一致性核验 + fresh-context Review」，合并决策留待用户。 |

---

## 1. Goal

在不改变现有 Microsoft Excel 功能与安装体验、且不修改任何现有 `ribbon/*.bas` 文件的前提下，为 XSTARS 交付一个独立的 Windows WPS 发行形态：WPS 官方 JS 加载项（Ribbon + JavaScript）+ 仅监听 `127.0.0.1` 的本地 Python 回环 HTTP 服务，完全离线运行，复用现有统计、预设、绘图、Tkinter 配置与设置持久化能力，以独立 `XSTARS_WPS_Setup.exe` 发布（现有 Excel 安装包保持不变）。

可验证结果：

1. WPS 端完成选区读取 → 分析 → 结果表写回 → 图片插入 → 高分辨率导出（PNG/TIFF/JPG/PDF，自定义 DPI），`XSTARS_Templates.xlsx` 的全部 Sheet（`Data`/`WB`/`qPCR`/`CCK8`/`ELISA`/`Transform`/`Standard Curve`）可在 WPS 中执行。
2. 数值结果与 Excel/Python 基准在算法容差内一致；图片嵌入、命名、定位不覆盖源数据/结果表，保存重开后仍存在。
3. 专业版阻断矩阵全绿；个人版形成 Beta 能力说明（失败项不阻断专业版首发）；Excel 现有模板 smoke test 与 142+ 自动化测试保持通过。
4. 所有 Milestone 在**验证闸门通过后才 commit/push 并进入下一 Milestone**（2026-08-30 决策 D-B）；未取得用户实机记录前，Draft PR 不转 Ready。

---

## 2. Requirements

### 2.1 硬约束

1. **R1 — 平台**：Windows 10/11 x64；首版阻断支持最新稳定版 WPS 365/12.x 专业版、商业版或政企版；WPS 个人版为 Beta/尽力支持，不因个人版专有问题阻断专业版首发。来源：旧文档 §2.1.1–2.1.3。
2. **R2 — 离线**：必须完全离线可用；仅允许本机回环 HTTP；允许安装 WPS JS 加载项与 `publish.xml`；无云服务/账号/遥测/外部网络调用。来源：旧文档 §2.1.6、§3。
3. **R3 — Excel 零回归**：保持现有 Excel 入口、Ribbon/VBA 回调、命令行调用与安装体验不回归；不改写现有统计方法、预设算法或绘图风格。来源：旧文档 §2.1.4、§3。
4. **R4 — 独立发行**：WPS 使用独立安装包 `XSTARS_WPS_Setup.exe`，不与 Excel 安装器合并。来源：旧文档 §2.1.5。
5. **R5 — WPS MVP 功能**：Run 与 Quick Run；WB、qPCR、CCK-8、ELISA；Transform Only 与 Standard Curve；主题/配色与 `~/.xstars/settings.json` 持久化；统计表、转换数据、图片写回；选中图片高分辨率导出 PNG/TIFF/JPG/PDF 自定义 DPI；结构化错误与诊断。来源：旧文档 §2.2。
6. **R6 — 安全边界**：服务仅绑定 `127.0.0.1`（测试必须证明未监听 `0.0.0.0`/局域网）；每安装实例随机密钥；Origin/CORS 最小白名单（以实测 `file://`/`null` 为准，不用 `Access-Control-Allow-Origin: *`）；命令枚举白名单，禁止任意 Python 函数名/shell 命令/输出路径穿越；受控临时目录与脱敏日志。来源：旧文档 §6.3。
7. **R7 — 服务拉起策略（M0.3 结论）**：加载项不得自行拉起本地服务；服务自启动由独立安装器负责（本计划 M6/T6.1）；加载项仅做健康检查与用户引导。来源：旧文档 M0.3 证据 + M0.3 结论。
8. **R8 — ELISA 二次选区（D-C）**：主验证 WPS JS API `Application.InputBox(..., Type=8)`（在 Ribbon 回调中调用，避开模态网页对话框拦截框选的坑）；两阶段 Ribbon 交互作为对照验证；地址输入（Tkinter）仅作两者均失败时的兜底，M0.4 不重复验证（M0.3 已验证 Tkinter 对话框可用）。来源：2026-08-30 访谈决策 D-C。
9. **R9 — 高清导出（D-D）**：主验证 JS `Shape.CopyPicture` → 本地服务读剪贴板（`PIL.ImageGrab`/`win32clipboard`）→ 按目标 DPI/格式重编码；对照探测 COM `Ket.Application` 可用性（`GetActiveObject` 一次性探测）；交付 PNG/TIFF/JPG/PDF × 至少 96/300/600 DPI 实测矩阵（含显示缩放 125%/150% 观察项）；记录 `CF_ENHMETAFILE` 矢量质量评估。来源：2026-08-30 访谈决策 D-D。
10. **R10 — 不引入 mac 重渲染依赖（D-E）**：不在 M0.4 引入 mac PR 的 `xstars/artifacts.py` 重渲染依赖；仅当剪贴板/COM 均不可行时，fallback 决策（含等 mac PR 合入后复用 `artifacts.py`）届时提交用户决定。来源：2026-08-30 访谈决策 D-E（进入待决事项 O3）。
11. **R11 — 测试风格（D-B）**：每个 Milestone 内嵌完整验证闸门（自动化测试 + 涉及宿主行为的实机验证）；测试与其实现同 Milestone 交付；**不设集中测试里程碑**；验证通过才允许 commit/push 并进入下一 Milestone。来源：2026-08-30 访谈决策 D-B。
12. **R12 — 旧文档处置（D-F）**：本 Plan 声明取代 `docs/wps-support-implementation-plan.md`；实施首个 Milestone 时给旧文档顶部加 superseded 横幅（指向本 Plan 路径），该动作列入文件级修改范围与任务。来源：2026-08-30 访谈决策 D-F。
13. **R13 — Git 策略（D-G）**：继续使用现有分支 `feature/wps-support` 与 Draft PR #1；本 Plan 文件作为重梳后首个提交（如 `docs(plan): restage wps support plan in 9-section format`）进入该分支；PR 标题不变；所有 Milestone 串行进入同一 PR；不采用 integration 分支。来源：2026-08-30 访谈决策 D-G。
14. **R14 — 实机验证责任（D-H）**：M0.4 代码就绪后用户即在真实 WPS 专业版 x64 断网环境执行验证清单；后续每个涉及宿主行为的 Milestone 同样嵌入实机验证闸门。来源：2026-08-30 访谈决策 D-H。
15. **R15 — 全范围重梳（D-A）**：M0.4 + 旧 M1–M6 → 共 7 个 Milestone 的完整 9 段式 Plan。来源：2026-08-30 访谈决策 D-A。

### 2.2 期望行为

1. **E1 — 分支工作流**：开发仅在 `feature/<short-name>` 分支；骨架可审阅即建 Draft PR；milestone 的实现/验证证据/风险/checklist 持续更新到同一 Draft PR；仅批准范围内功能完成、CI/实机验证通过、Review 无 Blocker 后转 Ready 并 squash merge。来源：旧文档 §2.3。
2. **E2 — 不静默缩小需求**：任意图片高清导出无法等价实现时，必须回到用户处决定是否接受「仅 XSTARS 生成图可重渲染」或专业版 COM fallback。来源：旧文档 §8.2。
3. **E3 — 错误/取消/诊断**：无效数据、取消对话框、服务未启动、端口冲突时 WPS 不崩溃且给出可行动错误。来源：旧文档 §8.3。
4. **E4 — 卸载安全**：卸载移除本产品加载项/服务文件，但不破坏用户其他 `publish.xml` 项或设置。来源：旧文档 §8.3。

### 2.3 已完成工作（继承；仅此小节保留 `[x]` 状态）

以下 M0.0–M0.3 来自旧计划，已通过实机验证。本重梳 Plan **不再占用 Milestone 名额**，仅作为证据基线继承。其余所有任务一律 `[ ]`。

- [x] **M0.0 创建功能分支并开启 Draft PR**
  - 文件：Git 分支、Draft PR 描述、`docs/wps-support-implementation-plan.md`
  - 验收：开发不发生在 `main`；Draft PR 可见。证据：Draft PR [#1](https://github.com/Frankkk1912/xstars/pull/1)。
  - 依赖：用户批准旧计划。

- [x] **M0.1 建立最小 WPS JS 加载项与官方离线部署 PoC**
  - 文件：`poc/wps/addin/*`、`poc/wps/README.md`
  - 验收：专业版断网可安装、显示 Ribbon、按钮回调、完全退出后重启回调存活、卸载；实际 Origin 为 `file://`。
  - 证据：WPS 365 教育高级版 `12.1.0.28022` 64 位完全断网；官方 `publish` 安装（`addinType=et`、`online=false`、`multiUser=false`、`version=1.1.0`）；记录见 Draft PR [#1](https://github.com/Frankkk1912/xstars/pull/1#issuecomment-5461289789)。
  - 依赖：M0.0。

- [x] **M0.2 验证 Selection/Value2/AddPicture 垂直链路**
  - 文件：`poc/wps/addin/*`、`poc/wps/probe_server.py`
  - 验收：保存并重开后数据和图片仍存在。
  - 证据：2026-08-29 完全断网实机，WPS 365 教育高级版 `12.1.0.28022` 64 位：`A1:B2` 二维 `Value2` 经 Origin `null` 预检发往 `127.0.0.1:3891/probe`，JSON 往返成功；`D1:E2` 写回矩阵一致；`Shapes.AddPicture` 嵌入 320×180 PNG（11.29 × 6.35 cm，`D1:E2` 下方）；另存副本、完全退出重开后值与图片均存在；全程无报错。实现锚点：`poc/wps/addin/js/ribbon.js:117-190`（`runM02Probe`）、`poc/wps/probe_server.py:79-113`（`validate_selection`）。
  - 依赖：M0.1。

- [x] **M0.3 验证服务拉起、CORS、Tkinter 与生命周期**
  - 文件：`poc/wps/*`
  - 验收：服务未运行时可恢复；WPS 不冻结；失败有可诊断结果。
  - 证据：2026-08-29 实机（宿主同前）。**关键负面发现**：`OAAssist.ShellExecute` 弹安全确认窗后静默不启动目标进程（入口日志 + 143 次采样无痕迹；官方 2 参数签名修正后依旧失效）→ 服务拉起策略改为 M5.1 安装器自启动（本计划 T6.1），加载项仅健康检查与引导。其余全过：真实 Origin `file://` 预检与 JSON 往返成功；Tkinter 对话框期间 WPS 不冻结且取消正确回报（`confirmed=False`）；`SO_EXCLUSIVEADDRUSE` 单实例、双开产生 `PORT CONFLICT` 日志并以退出码 2 退出、原服务不受影响；服务被杀后外部可恢复。实现锚点：`poc/wps/service_server.py:71-118`（`TkDialogManager`）、`127-140`（`Gate0ServiceServer`）、`344-421`（`self_test`）。
  - 依赖：M0.2。

> 注：Draft PR #1 body 中 M0.2/M0.3 checklist 尚未勾选（滞后于实际进度），重梳后首个 Milestone 的任务 T1.1 会修正 PR body。

---

## 3. Non-goals

首版明确不包含（继承旧文档 §3，并按 2026-08-30 决策补充，不扩大不缩小）：

- macOS WPS；WPS WebOffice、金山文档网页端或移动端。
- 32 位 WPS/Python。
- 重写现有统计方法、预设算法或绘图风格。
- 将 Tkinter 配置 UI 全面重写为 WPS 任务窗格。
- 将 Excel 端迁移到 WPS JS/HTTP 架构。
- 单一安装包自动安装 Excel 与 WPS 两套组件。
- 将非官方 VBA 补丁作为个人版依赖。
- 云服务、账号系统、遥测或外部网络调用。
- **M0.4 不验证** mac PR 的 `xstars/artifacts.py` Figure 重渲染导出路径（D-E；仅作 fallback 待决项）。
- **M0.4 不重复验证** Tkinter 地址输入交互（D-C；M0.3 已验证 Tkinter 对话框可用、不冻结、取消正确回报）。
- 不修改任何现有 `ribbon/*.bas`、`ribbon/customUI14.xml`、现有 Excel 安装包/制品。

---

## 4. Research summary

### 4.1 代码库现状与可复用模式（含行级锚点）

| 结论 | 证据 | 计划含义 |
| --- | --- | --- |
| Excel 主路径为 VBA RunPython/Shell → `xstars/cli.py` → `xstars/main.py` → xlwings 写回。 | `xstars/cli.py:1-36`（`main()` 命令分发）；`xstars/main.py:205-1507`（`run_*()` 入口族）。 | 复用 `DataHandler/Presets/StatsEngine/PlotEngine` 核心，不重写算法。 |
| 选区读取已宿主耦合。 | `xstars/data_handler.py:26-90`（`read_selection` / `read_selection_with_labels` / `read_from_range` 依赖 `Book.caller()`、`Range`）。 | 需新增从二维值/DTO 构造 DataFrame 的入口。 |
| 设置持久化与宿主无关。 | `xstars/config.py:12`（`DEFAULT_SETTINGS_PATH = Path.home()/".xstars"/"settings.json"`）；`config.py:164-218`（`save/load`）。 | WPS 主题/设置直接复用 JSON 持久化。 |
| Excel 侧 Shape 导出/二次选区依赖 COM。 | `xstars/main.py:881-894`（`_get_selected_shapes` 用 `Selection.ShapeRange`）、`897-942`（`_export_shape_highres` 用 `CopyPicture`+`PIL.ImageGrab`）、`1385-1413`（`_select_sample_data` 用 `InputBox(Type=8)`）。 | 这些是 WPS 缺口；M0.4 验证 WPS 等价路径，Excel 路径保持不变。 |
| Gate 0 PoC 已建立官方离线发布链路。 | `poc/wps/addin/scripts/build-publish-offline.cjs:1-118`（隔离 `publishlist.json`、断言 `online=false`/唯一条目）；`poc/wps/addin/package.json:1-19`（`wpsjs 2.2.3`、`npm test = node --test tests/*.test.cjs`）。 | M6 独立安装器复用该离线发布机制。 |
| Ribbon 回调与选区规整已稳定。 | `poc/wps/addin/ribbon.xml:1-36`（`onAction="OnAction"` 统一分发）；`poc/wps/addin/js/ribbon.js:77-108`（`normalizeSelectionValues` 处理标量/1D/2D）、`117-190`（`runM02Probe` 选区+写回+AddPicture）、`317-347`（`OnAction` 分发）。 | M0.4 探针按钮仿照该模式扩展。 |
| 回环服务安全纪律已建立。 | `poc/wps/probe_server.py:20-26`（`ALLOWED_ORIGINS = {"null","file://","http://127.0.0.1:3889"}`）、`138-251`（`ProbeRequestHandler` 路由/错误码）；`poc/wps/service_server.py:33-39`（同白名单）、`143-308`（`ServiceRequestHandler`）。 | M0.4 后端复用既有 CORS/日志/错误码/`SO_EXCLUSIVEADDRUSE`。 |
| 测试基线。 | `python -m pytest -q` → 142 passed, 3 warnings（旧文档 §4.2）；`tests/test_wps_probe.py:1-110`（Origin 拦截/预检/矩阵回显/PNG）；`poc/wps/addin/tests/ribbon.test.cjs:1-350`（`node:vm` mock WPS DOM/fetch）。 | 新测试沿用 pytest + `node:test` 双栈。 |

### 4.2 外部调研结论（来源、日期、置信度）

| 结论 | 来源、日期、置信度 | 采用方式 |
| --- | --- | --- |
| WPS ET JS API 原生支持 `Application.InputBox(..., Type=8)` 返回 Range；**避坑**：模态网页对话框（`modal:true`）会拦截框选，须在 Ribbon 回调/非模态对话框中调用。 | WPS 开放平台 Application.InputBox；WPS 社区 InputBox 模态冲突讨论；2025-05；置信度高。 | M0.4 主验证路径（R8/D-C）。 |
| `Shape.CopyPicture(Appearance, Format)` 原生支持（`xlScreen=1`/`xlPrinter=2`；`xlPicture=-4147`/`xlBitmap=2`）；`Chart.Export` 存在但纯 JS 导出仅屏幕 DPI（96~120）。 | WPS 开放平台 Shape.CopyPicture / Chart.Export；2025-05；置信度高。 | M0.4 导出主验证路径：JS `CopyPicture(2,-4147)` → 剪贴板 → Python 重编码（R9/D-D）。 |
| WPS Windows 版 COM ProgID 为 `Ket.Application`/`et.Application`；`pywin32 GetActiveObject("Ket.Application")` 可行，但 **UAC 完整性级别需一致**，否则 `-2147221005` 或权限拒绝；xlwings 不官方支持 WPS。 | WPS 开放平台 CreateObject/ProgID；GitHub xlwings #2281；2025-05；置信度高。 | M0.4 一次性对照探测（R9/D-D），非主路线。 |
| 混合架构「JS `CopyPicture` + 本地 Python 读剪贴板」极高可行，规避 COM 多实例句柄绑定。 | 本地调研综合结论；置信度高。 | M0.4 推荐架构。 |
| 剪贴板 DIB 直读受显示缩放（125%/150%）影响；600 DPI 印刷级需 `win32clipboard` 提取 `CF_ENHMETAFILE` 并用 GDI+/Pillow 渲染。 | 本地调研 Gaps 1；置信度中高。 | M0.4 实机矩阵纳入 125%/150% 观察项与 EMF 质量评估（R9）。 |

### 4.3 候选方案取舍（M0.4 形态）

- **ELISA 交互**：主选 InputBox(Type=8)（与 Excel 体验最一致），对照两阶段 Ribbon（carto 候选路径 1 的扩展点），兜底 Tkinter 地址输入（不重复实机验证）。依据：D-C + carto-report §5/§6。
- **导出**：主选「JS `CopyPicture` → 剪贴板 → Python 重编码」；对照一次性探测 COM `Ket.Application`；不采用纯 JS `Chart.Export`（仅屏幕 DPI，无法自定义 300/600）；不引入 `xstars/artifacts.py` 重渲染（D-E）。
- **M0.4 探针落点**：采用 carto 候选路径 1——扩展 `poc/wps/service_server.py` 与 `poc/wps/addin/js/ribbon.js` + `ribbon.xml`（复用既有 CORS/日志/单实例/测试体系），而非另建独立 `elisa_selection.py`/`shape_export.py` 多端口进程（候选路径 2，样板重复、端口管理成本高）。此选择改变旧计划 M0.4 的文件名（`poc/wps/elisa_selection.*`/`shape_export.*`），但**不改变其验证范围**。

### 4.4 证据缺口（显式标注）

1. `Application.InputBox(Type=8)` 在本宿主的真实返回对象/取消语义尚未实机验证（调研为文档+社区证据，置信度高但非实机）。
2. 选中 Shape 时 `window.Application.Selection` 的对象类型（是否含 `.ShapeRange`/`Selection.Type`）尚无实机证据。
3. `Shape.CopyPicture` → 剪贴板 → Python 重编码在 125%/150% 显示缩放下的 DPI 精度，以及 `CF_ENHMETAFILE` 矢量栅格化质量，无量化数据。
4. `Ket.Application` 的 `GetActiveObject` 在本机 UAC 完整性级别下是否可用，未实机确认。
5. 旧计划 M0.2/M0.3 的精确 commit SHA 未在本 Plan 提供（不虚构 SHA）；实施 T1.1 时从 PR #1 记录回填。

---

## 5. Gap analysis

| ID | 功能缺口 | 现状 | 影响 | 补齐任务 |
| --- | --- | --- | --- | --- |
| G1 | ELISA 二次选区在 WPS 无宿主交互实现 | 仅 `poc/wps/addin/js/ribbon.js:122-126` 一次性读当前选区；Excel 依赖 `xstars/main.py:1385-1413` 的 COM `InputBox(Type=8)` | ELISA/Standard Curve 反算无法在 WPS 完成 | T1.3、T1.4、T1.5、T1.6、T5.2 |
| G2 | 选中 Shape 识别 + 四格式/自定义 DPI 高清导出在 WPS 未验证 | `poc/wps` 无 Shape 选中/剪贴板/多格式重编码代码；Excel 依赖 `xstars/main.py:897-942` COM `CopyPicture`+`ImageGrab` | 导出功能在 WPS 可能不可用（高风险/可能阻断） | T1.3、T1.4、T1.5、T1.6、T5.4 |
| G3 | 服务生命周期未产品化 | M0.3 证明 `ShellExecute` 失效；服务仅能手工/外部拉起 | 加载项无法可靠恢复服务 | T3.1、T6.1 |
| G4 | 无宿主无关请求/响应契约 | 核心仍直接依赖 xlwings（`data_handler.py:26-90`、`main.py:205-1507`） | WPS broker 无法安全接入核心算法 | T2.2、T2.3 |
| G5 | Run/Quick/WB/qPCR/CCK8/Transform/Standard Curve 无 WPS 命令映射与写回 | 算法可复用，但宿主数据桥/WritebackPlan 缺失 | WPS 只能跑最小探针，无法完成产品流程 | T2.2、T2.3、T4.1、T4.2、T5.1、T5.2、T5.5 |
| G6 | 设置/主题/错误诊断无 WPS 入口与错误映射 | 设置 JSON 与宿主无关（`config.py:12`），但无 Ribbon 回调/稳定错误码 | WPS 用户无法改主题/读诊断 | T4.1、T5.3 |
| G7 | 本地 broker 安全边界缺失 | 现仅探针，无每实例密钥/命令白名单/请求限制/worker 管理 | 正式服务暴露本地数据与任意命令风险 | T3.1 |
| G8 | 单任务 worker + Tkinter GUI 主线程模型缺失 | M0.3 仅验证对话框线程（`service_server.py:71-118`） | Tkinter 需稳定 GUI 主线程，HTTP 线程直跑不安全 | T3.2 |
| G9 | 离线部署/独立安装器/卸载/用户文档缺失 | 仅开发期 `deploy/publish.html`；README 未声明 WPS | 无法交付、升级、卸载与复现断网安装 | T6.1、T6.2、T6.3 |
| G10 | 全模板阻断验收与个人版 Beta 矩阵缺失 | 无 `docs/wps-validation.md` 全矩阵 | 无法证明专业版全绿与个人版边界 | T7.1、T7.2、T7.3 |
| G11 | 计划治理与 PR 状态滞后 | Draft PR #1 body 中 M0.2/M0.3 未勾选；旧文档未被取代 | 审阅者与实施者对真实进度误判 | T1.1、T1.2 |
| G12 | 自动化测试/静态门禁未覆盖新层（横切） | 仅 142 个 Python 测试 + 既有 JS mock | 无法持续锁定契约/服务/JS 回归 | T1.5、T2.1、T2.2、T3.1、T3.2、T4.1、T4.2、T5.4 |

**Gap 自查**：G1–G12 每个缺口均至少映射一个稳定任务 ID；T1.1–T7.3 每个任务均支撑上表至少一个缺口（G12 为横切约束，其任务同时支撑 G1/G4/G5/G7/G8/G2）。**无孤立缺口、无范围外任务。**

---

## 6. Milestone 表格

| Milestone | Status | Dependencies | Validation | Notes |
| --- | --- | --- | --- | --- |
| M1 — Gate 0 收尾：M0.4 ELISA 二次选区 + 高清导出可行性 PoC + 治理 | [x] | 旧计划 M0.0–M0.3（已完成）；本 Plan rev 1 批准 | 自动化：`pytest -q` 全绿、`node --test` 全绿、`service_server.py --self-test` 通过、`git diff --check`；实机：用户执行 M0.4 清单（InputBox / 两阶段 / CopyPicture / 剪贴板 / 4 格式×96/300/600 DPI / 125%·150% 缩放 / 保存重开 / COM 探测 / EMF 评估）并记录 | 产出 M0.4 结论（选定交互与导出路径）；T1.1/T1.2 治理收尾；验证通过才进入 M2 |
| M2 — Excel characterization + application 契约 | [x] | M1 通过 | 自动化：142 基线 + 新契约/刻画/抽取测试全绿；实机：真实 Excel 模板 smoke test 无回归 | 先锁行为再抽取；对应旧 M1.1–M1.3 |
| M3 — 本地 broker + worker | [x] | M2 | 自动化：HTTP/安全/生命周期/worker 取消超时崩溃测试全绿；证明未暴露 `0.0.0.0`；旧 CLI 语法测试通过 | 服务自启动归安装器（T6.1）；对应旧 M2.1–M2.2 |
| M4 — WPS Ribbon + Run/Quick 垂直切片 | [x] | M3 | 自动化：`node --test` 全绿；实机：真实 Ribbon + Data Sheet Run/Quick + 保存重开 | 首个垂直切片；对应旧 M3.1–M3.2 |
| M5 — 预设/ELISA 落地/主题设置/高分辨率导出 | [x] | M4；M0.4 选定路径 | 自动化：预设/导出/设置测试全绿；实机：模板对应 Sheet 逐项 + 设置持久化 + 导出矩阵 | 采用 M0.4 选定交互与导出路径；对应旧 M4.1–M4.4 |
| M6 — 独立安装器/用户文档/离线加固 | [ ]（延后） | M5 | —— | **2026-08-31 用户决定：移出本 PR 范围**，延后到功能改进稳定后的独立 PR；本 PR 保持 Draft，交付物不含安装器 |
| M7 — Excel↔WPS 功能一致性核验 + fresh-context Review + 合并准备 | [x] | M5 | 功能一致性矩阵（Excel 入口 vs WPS 实现）+ 全量自动化 + Review 无 Blocker | 原「全模板阻断验收」中的安装器相关项随 M6 延后；合并决策留待用户。Review 三轮闭环：R1 1 Blocker+5 项 → R2 仅剩取消竞态 P1 → R3 generation 状态机修复（ff14449/308b6a4）；T5.6 实机复测通过（含导出另存为对话框 29680d0） |

Milestone 总数：**7**（=7，满足 ≤7 目标、≤10 上限）。所有初始 Status 均为 `[ ]`。

---

## 7. 分 milestone 的 To-do checkbox 清单

### M1 — Gate 0 收尾

- [x] **T1.1 更新 Draft PR #1 body：勾选 M0.2/M0.3、替换 checklist、链接新 Plan**
  - 文件：Draft PR #1 body（GitHub，非仓库文件）
  - 修改：勾选 M0.2/M0.3；把旧 checklist 替换为本 Plan 的 M1–M7 结构；链接 `plans/20260830-wps-support.md`；从 PR #1 记录回填 M0.2/M0.3 的精确 commit SHA。
  - 验收：PR body 反映真实进度；checklist 与新 Plan 一致；新 Plan 链接可达。
  - 依赖：本 Plan rev 1 批准；支撑 G11。

- [x] **T1.2 给旧文档顶部加 superseded 横幅**
  - 文件：`docs/wps-support-implementation-plan.md`（修改）
  - 修改：顶部加横幅「已被 `plans/20260830-wps-support.md` 取代，保留为历史证据」，指向新 Plan 路径。
  - 验收：横幅可见；旧文档正文保留；无内容删除。
  - 依赖：本 Plan rev 1 批准；支撑 G11、R12。

- [x] **T1.3 扩展 poc/wps 前端：M0.4 ELISA 选区探针（InputBox 主 + 两阶段对照 + 地址输入兜底代码）**
  - 文件：`poc/wps/addin/ribbon.xml`、`poc/wps/addin/js/ribbon.js`（修改）
  - 修改：新增 `<group id="xstarsM04Group">` 与按钮；新增 `runM04InputBoxProbe`（Ribbon 回调中 `window.Application.InputBox(..., 8)` 取 Range）、`runM04TwoStageProbe`（两阶段按钮记录标准品/样本选区，内存暂存 + 状态回显）、地址输入兜底函数（仅代码，不重复实机验证）；复用 `normalizeSelectionValues`/`fetchService`/`OnAction` 分发模式。
  - 验收：`node --test` 新增用例覆盖 InputBox mock 返回 Range、两阶段状态机、地址兜底函数可解析；既有 M0.2/M0.3 用例不回归。
  - 依赖：T1.1；支撑 G1、G12、R8。

- [x] **T1.4 扩展 poc/wps 后端：ELISA 选区端点 + Shape/剪贴板导出端点 + COM 一次性探测**
  - 文件：`poc/wps/service_server.py`（修改；复用候选路径 1，替代旧计划的独立 `elisa_selection.py`/`shape_export.py`）
  - 修改：新增 `/probe/elisa-selection`（复用 `probe_server.validate_selection` 的矩形/边界校验，接收两阶段或 InputBox 的选区 payload）；新增 `/probe/shape-export`（JS `CopyPicture(2,-4147)` 后，服务端用 `PIL.ImageGrab.grabclipboard()`/`win32clipboard` 读 DIB/EMF，按 `{format, dpi}` 重编码 PNG/TIFF/JPG/PDF）；新增 `/probe/com-probe`（一次性 `win32com.client.GetActiveObject("Ket.Application")`，记录成功/失败码）；严格沿用 `ALLOWED_ORIGINS`/`SO_EXCLUSIVEADDRUSE`/JSON 错误码。
  - 验收：`service_server.py --self-test` 扩展通过；`pytest` 覆盖端点契约与错误码；COM 探测在无 WPS 环境返回可诊断错误而非崩溃。
  - 依赖：T1.1；支撑 G1、G2、G12、R9。

- [x] **T1.5 双栈自动化测试扩展（pytest + node:test）**
  - 文件：`tests/test_wps_probe.py`（修改/扩展）、`poc/wps/addin/tests/ribbon.test.cjs`（修改/扩展）；必要时新建 `tests/test_wps_export_probe.py`
  - 修改：Python 覆盖 `/probe/elisa-selection` 边界、`/probe/shape-export` 格式/DPI 参数校验与剪贴板不可用时的错误、`/probe/com-probe` 容错；JS 覆盖 InputBox/两阶段/地址兜底 mock。
  - 验收：`python -m pytest -q` 与 `node --test` 全绿；测试不依赖真实 WPS/剪贴板/COM。
  - 依赖：T1.3、T1.4；支撑 G1、G2、G12。

- [x] **T1.6 实机执行 M0.4 验证清单并回填证据（责任人：用户）**
  - 文件：`poc/wps/README.md`（追加 M0.4 记录）、Draft PR #1（评论/body）
  - 修改：执行 M0.4 清单——InputBox(Type=8) 返区/取消；两阶段 Ribbon 对照；地址兜底仅确认代码存在；选中 Shape 的 `Selection` 对象类型；`CopyPicture` → 剪贴板 → Python 重编码；PNG/TIFF/JPG/PDF × 96/300/600 DPI；125%/150% 显示缩放观察；保存重开；`Ket.Application` COM 探测；`CF_ENHMETAFILE` 矢量质量评估。
  - 验收：产出明确结论（ELISA 选定交互路径；导出选定路径或触发 O3 fallback 决策）；全部证据（版本/截图/文件哈希）回填 PR。
  - 依赖：T1.3–T1.5；支撑 G1、G2、R8、R9、R14。

### M2 — Excel characterization + application 契约

- [x] **T2.1 增加 Excel 行为刻画测试**
  - 文件：`tests/test_end_to_end.py`（修改/扩展）或新增 `tests/test_excel_characterization.py`
  - 修改：锁定现有入口（`run`/`run_quick`/`run_wb`/`run_qpcr`/`run_cck8`/`run_elisa`/`run_transform_only`/`run_standard_curve`/`run_export`）、选区解析、结果表位置、图片命名、错误、导出与取消行为。
  - 验收：重构前可稳定复现当前行为；142 基线全绿。
  - 依赖：M1 通过；支撑 G12、G4、G5。

- [x] **T2.2 定义宿主无关请求/响应契约**
  - 文件：`xstars/application/contracts.py`（新建）、`tests/test_application_contracts.py`（新建）
  - 修改：命令白名单、`SelectionPayload`、`WritebackPlan`、`Artifact` DTO、稳定错误码、版本字段；序列化往返/非法输入/路径与大小边界校验。
  - 验收：契约单测通过；命令枚举闭合；无任意函数名/shell/路径穿越。
  - 依赖：T2.1；支撑 G4、G5、G12、R6。

- [x] **T2.3 抽取共享 application 用例并保持 Excel 回归**
  - 文件：`xstars/application/analysis.py`（新建）、`xstars/main.py`（修改）、`xstars/data_handler.py`（修改）、`tests/test_application_analysis.py`（新建）
  - 修改：将计算/产物生成与 xlwings 写回分离；`data_handler.py` 新增从二维值/DTO 构造 DataFrame 入口；`main.py` 委托 application 层但保留全部公开入口与 Excel 写回行为不变。
  - 验收：142+ 测试全绿；真实 Excel 模板 smoke test 无回归。
  - 依赖：T2.1、T2.2；支撑 G4、G5、G12、R3。

### M3 — 本地 broker + worker

- [x] **T3.1 实现本地 broker 安全边界**
  - 文件：`xstars/wps_service.py`（新建）、`tests/test_wps_service.py`（新建）
  - 修改：仅 `127.0.0.1` 监听（测试证明未暴露 `0.0.0.0`）；每安装实例密钥；Origin 白名单；命令白名单；请求体大小/并发限制；`/health`；单实例（`SO_EXCLUSIVEADDRUSE`）与端口范围；服务不负责自启动（归 T6.1）。
  - 验收：安全/错误/端口/并发/鉴权测试全绿。
  - 依赖：M2 通过；支撑 G3、G7、G12、R6、R7。

- [x] **T3.2 实现单任务 worker 与 GUI 主线程模型**
  - 文件：`xstars/application/worker.py`（新建）、`xstars/cli.py`（修改，新增受限 `serve`/`worker` 模式）、相关测试
  - 修改：broker 启动受控 worker 子进程；worker 在主线程跑 Tkinter；原子写请求/结果文件；取消、超时、崩溃恢复、临时文件清理；旧 CLI 语法 `<command> <workbook_path>` 不变。
  - 验收：Run/Quick mock E2E；Tkinter 取消无残留；旧 CLI 语法回归通过。
  - 依赖：T3.1；支撑 G8、G12、R3。

### M4 — WPS Ribbon + Run/Quick 垂直切片

- [x] **T4.1 实现 WPS Ribbon 与服务客户端**
  - 文件：`wps-addon/ribbon.xml`、`wps-addon/main.js`、`wps-addon/service-client.js`、`wps-addon/config.template.js`、`wps-addon/assets/*`（新建）
  - 修改：复刻批准范围内 Ribbon；服务探测/健康检查/鉴权（token）/错误展示；不依赖 `imageMso`（自有图标）；`config.template.js` 由安装器注入端口范围与每实例密钥。
  - 验收：`node --test` 覆盖服务发现/鉴权/错误映射；实机 Ribbon 显示与 Run/Quick 可用。
  - 依赖：M3 通过；支撑 G5、G6、G12、R6。

- [x] **T4.2 实现 WPS Selection 与 WritebackPlan 执行器**
  - 文件：`wps-addon/spreadsheet.js`（新建）、`wps-addon/tests/*`（新建）
  - 修改：连续选区验证、`Value2` 序列化、批量写回、图片命名/定位、状态栏与错误映射（复用 `poc/wps` 的规整语义）。
  - 验收：Data Sheet Run/Quick 完整垂直切片（实机）+ 保存重开结果不丢失。
  - 依赖：T4.1；支撑 G5、G12。

- [x] **T4.3 实机验证 Run/Quick 垂直切片（责任人：用户）**
  - 文件：Draft PR #1（记录）；无代码文件新增
  - 修改：真实 WPS 专业版 x64 断网执行 Data Sheet Run/Quick + 保存重开 + 取消/错误用例。
  - 验收：切片通过或发现明确阻断并回填。
  - 依赖：T4.2；支撑 G5、R14。

### M5 — 预设/ELISA 落地/主题设置/高分辨率导出

- [x] **T5.1 接通 WB/qPCR/CCK-8、Transform 与 Standard Curve**
  - 文件：`xstars/application/analysis.py`（修改）、`wps-addon/main.js`（修改）、相关测试
  - 修改：增加各命令映射与特定 WritebackPlan；核心算法不变。
  - 验收：模板对应 Sheet 在专业版逐项通过，数值与 Python/Excel 基准一致。
  - 依赖：M4 通过；支撑 G5。

- [x] **T5.2 完成 ELISA 交互落地（采用 M0.4 选定路径）**
  - 文件：`xstars/application/analysis.py`（修改）、`wps-addon/main.js`、`wps-addon/spreadsheet.js`（修改）、相关测试
  - 修改：实现 M0.4 选定的标准数据/样本数据选择协议与取消恢复。
  - 验收：ELISA 模板完整拟合、反算、统计、图表与可选标准曲线通过。
  - 依赖：T5.1、M0.4 结论（M1）；支撑 G1、G5、R8。

- [x] **T5.3 完成主题/设置与错误诊断**
  - 文件：`wps-addon/ribbon.xml`、`wps-addon/main.js`（修改）、`xstars/config.py`（修改，仅向后兼容字段）、`xstars/wps_service.py`（修改）
  - 修改：主题命令、设置共享、稳定错误码、脱敏日志与诊断包；`serve` 启动时将 port 持久化到 `~/.xstars/wps_service.json`，inject-config 默认从该文件读取端口（T4.3 实机发现：注入端口与 broker 实际端口错位导致连接失败）
  - 验收：设置跨 WPS 重启持久化；故障用例不崩溃且日志可定位。
  - 依赖：T5.1；支撑 G6、R5、R6。

- [x] **T5.4 完成高分辨率导出（2026-08-31 用户定案方案 A：重渲染主路径 + 剪贴板 bonus）**
  - 文件：`xstars/application/export.py`（新建）、`tests/test_wps_export.py`（新建）、`wps-addon/main.js`（修改）、`xstars/application/worker.py`（修改）、`xstars/application/analysis.py`（修改）
  - 修改：生成图表时 best-effort 持久化轻量重渲染 payload（清洗数据+配置快照，`~/.xstars/artifacts/`，由 worker 下发 pictureId、加载项重命名 Shape 关联，不阻断出图）；导出命令对 XSTARS 图加载 payload 重渲染目标 DPI（主路径，真细节）；任意选中图片走剪贴板重编码（bonus）；payload 缺失/损坏给出可诊断错误；严格校验格式、DPI 与目标路径。
  - 验收：XSTARS 图导出 PNG/TIFF/JPG/PDF 为重渲染输出（独立验证 DPI、像素与文件有效性）；非 XSTARS 图走剪贴板路径；DPI 按请求值精确。
  - 依赖：T5.1、M0.4 结论（M1）、M4；支撑 G2、G12、R9。

- [x] **T5.5 实机验证预设/导出/设置（责任人：用户）**
  - 文件：Draft PR #1（记录）；无代码文件新增
  - 修改：模板对应 Sheet 逐项、设置持久化、导出矩阵（4 格式 × 96/300/600 DPI）。
  - 验收：逐项通过或有获批豁免并回填。
  - 依赖：T5.1–T5.4；支撑 G5、G6、R14。

### M6 — 独立安装器/用户文档/离线加固

- [ ] **T6.1 构建 WPS 独立可重复安装包**
  - 文件：`installer/wps/xstars-wps.spec`、`installer/wps/XSTARS_WPS.iss`、`installer/wps/build.ps1`（新建）
  - 修改：打包 service/worker/add-in；生成每安装实例密钥并注入 `config.template.js`；按官方机制安装/升级/卸载加载项；**服务自启动由安装器编排**；用户级目录优先；备份并恢复其修改的加载项配置。
  - 验收：干净 Win10/11 x64 可安装、升级、卸载；不影响 Excel 或其他 WPS 加载项。
  - 依赖：M5 通过；支撑 G3、G9、R7。

- [ ] **T6.2 完善用户文档与版本声明**
  - 文件：`docs/wps-installation.md`（新建）、`README.md`、`README.zh-CN.md`（修改）
  - 修改：记录专业版正式支持、个人版 Beta、版本基线、离线安装、故障诊断、卸载。
  - 验收：文档步骤在干净测试机可复现；不夸大个人版支持。
  - 依赖：T6.1；支撑 G9。

- [ ] **T6.3 实机验证安装/升级/卸载（责任人：用户）**
  - 文件：Draft PR #1（记录）；无代码文件新增
  - 修改：干净 Win10/11 x64 执行安装、Ribbon 显示、服务自启动、升级、卸载与配置恢复。
  - 验收：通过并回填证据。
  - 依赖：T6.1、T6.2；支撑 G9、R14。

### M7 — 全模板阻断验收 + fresh-context Review + 合并准备

- [ ] **T7.1 执行专业版阻断矩阵与个人版 Beta 矩阵（责任人：用户）**
  - 文件：`docs/wps-validation.md`（新建）
  - 修改：逐 Sheet、逐功能、逐系统记录版本/命令/截图/文件哈希与结果。
  - 验收：专业版全部阻断项通过；个人版差异明确记录。
  - 依赖：M6 通过；支撑 G10、R14。

- [ ] **T7.2 完成 fresh-context Review、修复与最终 diff 检查**
  - 文件：所有变更文件、Draft PR
  - 修改：并行审查正确性/回归、测试、可维护性，并增加安全、用户流程与安装/文档契约审查；仅由单一 fix worker 处理「现在值得修复」。
  - 验收：无 Blocker 或立即值得修复项；最多三轮；父 Agent 检查最终 diff。
  - 依赖：T7.1；支撑 G10。

- [ ] **T7.3 合并准备（转 Ready 条件确认；合并本身留待用户）**
  - 文件：Draft PR #1（无代码变更）
  - 修改：确认全部门禁通过、VBA/Excel 零 diff、无未批准任务被错误标记完成；准备转 Ready 与 squash merge 说明。
  - 验收：转 Ready 条件清单逐项满足；**合并动作由用户决定并执行**。
  - 依赖：T7.2；支撑 G10、R13。

---

## 8. Validation contract

只有满足对应 Milestone 合同，任务才允许从 `[ ]` 更新为 `[x]`。

### 8.1 自动化检查（每个 Milestone 的公共闸门）

| 检查项 | 命令或验证方式 | 预期结果 | 通过标准 | 责任人/限制 |
| --- | --- | --- | --- | --- |
| Python 基线 + 新增 | `python -m pytest -q` | 现有 142 测试全过；允许既有 3 个 warning，无新增未解释 warning | 退出码 0，0 failed/0 errors | 实施者；CI 可执行 |
| 契约/服务/导出专项 | `python -m pytest tests/test_application_contracts.py tests/test_application_analysis.py tests/test_wps_service.py tests/test_wps_export.py -v`（按里程碑分阶段） | 全过 | 0 failed | 实施者；CI 可执行 |
| JS 单元测试 | `cd poc/wps/addin && npm test`；正式 `wps-addon` 同理（`node --test tests/*.test.cjs`） | 选区/写回/错误映射/服务发现/InputBox/两阶段/CopyPicture mock 全过 | npm 退出码 0 | 实施者；需 Node 环境 |
| 服务自检 | `python poc/wps/service_server.py --self-test` | `SELF-TEST PASSED` | 退出码 0 | 实施者 |
| 静态语法 | `python -m compileall -q xstars poc tests` | 无 SyntaxError | 退出码 0 | 实施者；仓库无 linter 配置 |
| 空白/冲突标记 | `git diff --check` | 无 trailing whitespace/conflict marker | 无输出，退出码 0 | 实施者 |
| VBA/Excel 零修改 | `git diff --exit-code origin/main...HEAD -- 'ribbon/*.bas' 'ribbon/customUI14.xml'` | 与基线一致 | 无 diff，退出码 0 | 实施者 + reviewer；若 base 分支名不同替换为实际 base |
| 变更范围 | `git diff --name-status origin/main...HEAD` | 仅出现第 9 节允许文件 | 无未批准文件、无删除 | 实施者 + reviewer |
| 无 staged 临时产物 | `git diff --cached --exit-code` + `git status --short` 审阅 | 无意外 staged/未跟踪生成物 | 无 artifact/图片/缓存/临时 payload 进入提交 | 实施者 |
| 离线构建冒烟 | `cd poc/wps/addin && npm run wps:package:offline`（M6 及 M1 复用） | `publishlist` 仅含本插件、`online=false`、唯一条目 | 脚本断言通过 | 实施者；仅开发机冒烟，不提交生成物 |

### 8.2 实机验证清单（责任人均为**用户**，真实 WPS 专业版 x64、完全断网）

**M1（M0.4，对应 T1.6）**：

- [ ] `Application.InputBox(..., Type=8)` 返回 Range，可提取 `Address()`；取消行为可区分；
- [ ] 两阶段 Ribbon 对照：先选标准品、再选样本，状态回显正确；
- [ ] 地址输入兜底：确认代码存在（不重复实机验证）；
- [ ] 选中 Shape 时记录 `window.Application.Selection` 对象类型；
- [ ] `Shape.CopyPicture(2, -4147)` → 本地服务读剪贴板成功；
- [ ] PNG/TIFF/JPG/PDF × 至少 96/300/600 DPI 实测矩阵；
- [ ] 125%/150% 显示缩放下的 DPI 精度观察项；
- [ ] `Ket.Application` COM `GetActiveObject` 一次性探测（记录成功/失败码）；
- [ ] `CF_ENHMETAFILE` 矢量质量评估记录；
- [ ] 保存重开数据/图片不丢失。

**M2**：真实 Excel 模板 smoke test（证明现有安装/入口未变化）。

**M4**：真实 Ribbon + Data Sheet Run/Quick + 保存重开。

**M5**：模板对应 Sheet（WB/qPCR/CCK8/ELISA/Transform/Standard Curve）逐项 + 设置持久化 + 导出矩阵。

**M6**：干净 Win10/11 x64 安装/升级/卸载；服务自启动；不影响其他加载项配置。

**M7**：专业版阻断矩阵全绿 + 个人版 Beta 能力报告。

无法自动验证的原因：CI 不含真实 WPS GUI、`publish` 安装、剪贴板、COM UAC 交互、显示缩放与宿主行为；故以用户实机记录为合并 blocker（R14）。

### 8.3 Reviewer gate

- Reviewer 必须核对：Excel 路径是否原样保留、broker 是否仅回环 + token + 命令白名单、worker 是否隔离 GUI、`wps-addon` 是否不依赖 `imageMso`、导出是否满足 4 格式 × 96/300/600 DPI、全部生成产物是否不进入提交、文档是否不夸大个人版支持。
- Blocker 规则：任一 `ribbon/*.bas`/`customUI14.xml` 变更、任一 Excel 行为回归、broker 未限制回环/无鉴权、高清导出静默降级（见 R/E2）、专业版阻断项未通过、缺少用户实机记录，均阻止转 Ready。

---

## 9. 文件级修改范围 + 风险 / 回滚 / 待决事项 + Git 策略

### 9.1 文件级修改范围

**新建**

| 路径 | 用途 |
| --- | --- |
| `plans/20260830-wps-support.md` | 本重梳 Plan |
| `tests/test_application_contracts.py`、`tests/test_application_analysis.py`、`tests/test_wps_service.py`、`tests/test_wps_export.py` | 契约/用例/服务/导出测试 |
| `tests/test_wps_export_probe.py`（如需） | M0.4 导出端点专项测试 |
| `xstars/application/contracts.py` | SelectionPayload/命令枚举/WritebackPlan/Artifact DTO/错误码 |
| `xstars/application/analysis.py` | 宿主无关 Run/Quick/预设/Transform/Standard Curve 用例 |
| `xstars/application/export.py` | 导出命令、格式/DPI 校验、剪贴板/COM 协调 |
| `xstars/application/worker.py` | 受控请求文件执行、Tkinter 主线程、原子写结果 |
| `xstars/wps_service.py` | 127.0.0.1 broker、鉴权、CORS、单实例、worker 管理、健康检查 |
| `wps-addon/ribbon.xml`、`wps-addon/main.js`、`wps-addon/service-client.js`、`wps-addon/spreadsheet.js`、`wps-addon/config.template.js`、`wps-addon/assets/*`、`wps-addon/package.json`、`wps-addon/tests/*` | 正式 WPS 加载项 |
| `wps-addon/index.html`、`wps-addon/manifest.xml`、`wps-addon/vite.config.js`、`wps-addon/scripts/inject-config.cjs`、`wps-addon/scripts/build-offline-publish.cjs`（2026-08-31 批准范围扩展） | 官方 wpsjs 加载项壳与离线发布工具链（适配自 PoC 同名脚本，零业务逻辑）；缺失则无法打包安装进行 T4.3 实机验证 |
| `installer/wps/xstars-wps.spec`、`installer/wps/XSTARS_WPS.iss`、`installer/wps/build.ps1` | WPS 独立安装包 |
| `docs/wps-installation.md`、`docs/wps-validation.md` | 用户文档 + 实机矩阵模板 |

**修改**

| 路径 | 修改 |
| --- | --- |
| `docs/wps-support-implementation-plan.md` | 顶部加 superseded 横幅（T1.2） |
| `poc/wps/addin/ribbon.xml`、`poc/wps/addin/js/ribbon.js` | 新增 M0.4 ELISA/Shape 探针按钮与逻辑（T1.3） |
| `poc/wps/service_server.py` | 新增 `/probe/elisa-selection`、`/probe/shape-export`、`/probe/com-probe`（T1.4） |
| `poc/wps/README.md` | 追加 M0.4 实机验收记录（T1.6） |
| `tests/test_wps_probe.py` | 扩展新端点契约测试（T1.5） |
| `poc/wps/addin/tests/ribbon.test.cjs` | 扩展 InputBox/两阶段/地址兜底 mock（T1.5） |
| `tests/test_end_to_end.py`（或新增 characterization 文件） | Excel 行为刻画（T2.1） |
| `xstars/main.py` | 分阶段委托 application 层；保留全部公开入口与 Excel 写回行为（T2.3） |
| `xstars/data_handler.py` | 新增从二维值/DTO 构造 DataFrame 入口；保留现有 xlwings 方法（T2.3） |
| `xstars/cli.py` | 新增受限 `serve`/`worker` 模式；旧语法不变（T3.2） |
| `xstars/config.py` | 如需 WPS service 配置，仅向后兼容字段；统计/主题设置格式不破坏（T5.3） |
| `pyproject.toml` | 如需打包入口/依赖，仅向后兼容；优先标准库服务，避免无必要 Web 框架 |
| `README.md`、`README.zh-CN.md` | 验收后记录 WPS 专业版正式/个人版 Beta（T6.2） |

**明确不修改**

- `ribbon/customUI14.xml`、`ribbon/*.bas`（含 `ribbon_callbacks.bas`、`ribbon_callbacks_installed.bas`、`ribbon_callbacks_standalone.bas`）；
- 现有 Excel 安装包/制品（`XSTARS_Setup_v1.1.1.exe`、`xstars_v1.1.1.zip`）；
- `xstars/stats_engine.py`、`xstars/plot_engine.py`、`xstars/styles.py`、`xstars/annotations.py`；
- `xstars/presets/*`、`xstars/tools/*` 的算法（除非测试证明现有宿主耦合且变更不改变算法结果）；
- `XSTARS_Templates.xlsx`；
- 不删除任何文件。

### 9.2 风险

| 风险 | 等级 | 触发条件 | 影响 | 缓解 |
| --- | --- | --- | --- | --- |
| 任意 Shape 四格式/自定义 DPI 等价导出不可行 | 高/阻断 | 剪贴板 DIB 受显示缩放影响、EMF 栅格化精度不足、COM 不可用 | 导出功能无法等价交付 | M0.4 前置；主验剪贴板重编码 + 对照 COM 探测；失败触发 O3 fallback 决策，不静默缩小需求 |
| InputBox(Type=8) 实机行为与文档不符 | 高 | 返回非 Range、取消语义异常、模态拦截 | ELISA 二次选区无可用交互 | 主验 + 两阶段对照 + Tkinter 兜底（M0.3 已验证） |
| UAC 完整性级别不一致致 COM `GetActiveObject` 失败 | 中 | 服务与 WPS 权限级别不同 | COM 对照探测失败 | 仅作对照探测；主路线不依赖 COM；记录失败码 |
| 个人版不能按官方 publish 方式安装加载项 | 高（非阻断） | 个人版实测安装失败 | 个人版不可用 | Beta 定位；失败不阻断专业版首发，记录能力矩阵 |
| 手写 `publish.xml` 与官方格式不一致 | 高 | 直接手写发布配置 | 安装失败 | 只以官方 `wpsjs` 生成物与真实安装流程为准 |
| Tkinter 从 HTTP 线程调用不安全/焦点丢失 | 高 | broker 内直接建 GUI | WPS 冻结/崩溃 | broker + 独立 worker；GUI 在 worker 主线程；M0.3 已验不冻结 |
| WPS JS Origin/CORS 与 Chromium 假设不同 | 中高 | 未来 build 改变 Origin | 请求被 403 | 最小白名单（实测 `file://`/`null`）；不用通配符 |
| `ShellExecute` 被个人版/EDR 阻止 | 中高 | 加载项拉起服务 | 服务无法恢复 | 已转安装器自启动（M0.3 结论）；加载项仅健康检查/引导 |
| 多实例 WPS/并发点击造成错误写回 | 中 | 双开/连点 | 写错单元格 | 请求携带 workbook/Sheet/选区快照 + job id；单任务锁；写回前检查上下文 |
| 临时图/日志泄露实验数据 | 中 | 长驻缓存 | 隐私风险 | 受控目录、最短保留、脱敏日志、退出/启动清理 |
| 未签名 PyInstaller/Inno Setup 被 EDR 误报 | 中 | 政企环境预检 | 安装被拦 | 干净机预检；发布前决定代码签名；提供校验和 |
| 显示缩放（125%/150%）下 DPI 精度不达标 | 中 | 高 DPI 显示器 | 导出像素/DPI 元数据偏差 | M0.4 实机矩阵显式观察；必要时 EMF→GDI+ 渲染 |

### 9.3 回滚

1. **M0.4 失败**：停用/删除 M0.4 探针按钮与端点，回到 M0.3 状态；不修改 Excel/核心；重新决策架构（COM-only 或桌面应用方向）。
2. **M2 回归**：application 抽取保持小提交/小 diff；未通过 characterization 前不切换 Excel 路径；Excel 入口保留旧路径。
3. **M3/M4 失败**：停用 WPS service/add-in；共享核心与 Excel 路径仍可独立发布。
4. **M5 单功能失败**：不得把未完成阻断功能标记完成；高清导出或 ELISA 无法满足时暂停并请求用户决策。
5. **安装器失败**：保留开发/便携验证流程；不发布 Setup，不触碰现有 Excel 安装包。
6. **发布回滚**：WPS 独立版本可单独卸载/回退；安装器必须备份并恢复其修改的加载项配置，不删除其他产品条目。
7. **数据兼容性**：本方案不修改工作簿格式、不写 Excel picture metadata、不改变 `~/.xstars/settings.json` 既有语义；WPS 临时产物为受控缓存，删除后不影响用户数据。

### 9.4 待决事项（未获批准前实施者不得拍板）

1. **O1 — ELISA 最终交互路径**：InputBox(Type=8) vs 两阶段 Ribbon vs 地址输入，待 M0.4（T1.6）结论定案。
2. **O2 — 高清导出最终路径**：JS 剪贴板 vs COM `Ket.Application` vs 组合，待 M0.4 结论定案。
3. **O3 — 剪贴板/COM 均不可行时的 fallback**（D-E）：是否接受「仅 XSTARS 生成图可重渲染」或等 mac PR 合入后复用 `artifacts.py`；届时提交用户决定。
4. **O4 — 服务端口机制**：固定端口 vs 端口范围扫描 vs 官方配置注入，待 M3 契约设计定案。
5. **O5 — 安装包代码签名**：若目标 EDR 环境要求，将成为发布前阻断项。
6. **O6 — 个人版精确 build 与加载项能力差异**：待 M7 Beta 矩阵记录。
7. **O7 — 官方离线 publish 包的真实目录/配置格式**：以 M0.1 生成物为准，不在实现前假定手写 XML。
8. **O8 — 旧文档完整处置**：当前决定为「保留正文 + 顶部横幅」（R12/D-F）；是否后续归档删除需另行确认。

### 9.5 Git 策略

- **分支名**：`feature/wps-support`（沿用，不新建）。
- **Draft PR 标题**：`feat: add Windows WPS support`（沿用，不变）。
- **PR 拆分决策**：单个 Draft PR #1，M1–M7 串行进入同一 PR，不拆 PR、不建 integration 分支。依据（D-G）：Milestone=7 且为单一子系统方向，不满足升级条件；拆分会造成不可用中间态与重复验收。
- **合并顺序**：本 Plan 文件作为重梳后首个提交进入分支（建议提交信息 `docs(plan): restage wps support plan in 9-section format`），随后 M1 → M2 → … → M7 串行 commit/push；每个 Milestone 验证通过后才 commit/push 并进入下一 Milestone；全部通过 + Review 无 Blocker + 用户实机记录齐全后，由用户执行转 Ready 与 squash merge（合并本身留待用户）。
- **Draft PR 描述草稿**：

```markdown
## Summary

Adds an independent Windows WPS distribution for XSTARS: an official WPS JS
add-in (Ribbon + JavaScript) plus a loopback-only (127.0.0.1) local Python
service, fully offline, reusing the existing stats/preset/plot/config core and
shipped as a standalone XSTARS_WPS_Setup.exe. The existing Excel installer and
behavior are unchanged.

## Scope

- Windows 10/11 x64; blocking target = latest stable WPS 365/12.x
  Professional/Business/Government editions; WPS Personal = Beta (best effort).
- MVP: Run / Quick Run / WB / qPCR / CCK-8 / ELISA / Transform Only /
  Standard Curve; theme & settings persistence; writeback; selected-picture
  high-res export (PNG/TIFF/JPG/PDF, custom DPI).
- Service launch: installer-autostart only; the add-in performs health checks
  and guidance (Gate 0 found OAAssist.ShellExecute silently fails).

## Excel regression boundary

- All existing ribbon/*.bas and customUI14.xml are byte-for-byte unchanged.
- Existing Excel entry points, CLI syntax and installer behavior are unchanged.

## Design

WPS add-in reads Selection.Value2/Address, POSTs a JSON command to a
127.0.0.1 broker (per-install secret, minimal Origin allowlist, command
enumeration), the broker runs a single-task worker (Tkinter on the worker main
thread) that calls the host-agnostic xstars/application layer and returns a
WritebackPlan; the add-in writes Value2 back and embeds pictures via
Shapes.AddPicture. High-res export uses JS Shape.CopyPicture -> local service
reads the clipboard -> re-encode at the target DPI/format (M0.4 primary path).

## Validation

- [ ] python -m pytest -q (142 baseline + new tests)
- [ ] node --test for poc/wps/addin and wps-addon
- [ ] service_server.py --self-test
- [ ] git diff --check; VBA/Excel zero-diff gate
- [ ] M0.4 real-machine checklist (InputBox / two-stage / CopyPicture /
      clipboard / 4 formats x 96/300/600 DPI / 125%·150% scaling / COM probe)
- [ ] M4/M5/M6/M7 real-machine gates

## Risks / rollback

Key risks: arbitrary-shape 4-format custom-DPI export equivalence, ELISA
second-selection UX, UAC integrity-level for COM, display-scaling DPI
precision. The feature can be reverted as one PR; WPS artifacts are controlled
derived cache and do not modify workbook source data or ~/.xstars/settings.json.

## Governance

This plan supersedes docs/wps-support-implementation-plan.md (banner added).
Merging is left to the user; the PR stays Draft until real-machine records are
complete.
```

### 9.6 落盘前自查

- [x] 9 段结构齐全且顺序固定：Goal / Requirements / Non-goals / Research summary / Gap analysis / Milestone 表格 / To-do / Validation contract / 文件级范围+风险/回滚/待决+Git 策略。
- [x] Milestone 数量 = 7，≤7 目标且 ≤10 上限；所有初始 Status 为 `[ ]`；M0.0–M0.3 历史放 §2.3 不占名额。
- [x] G1–G12 每个缺口至少映射一个任务 ID；所有 T1.1–T7.3 均支撑至少一个缺口：**无孤立缺口、无范围外任务**。
- [x] 每个任务含 文件/修改/验收/依赖 四要素。
- [x] Validation contract 含命令/预期/通过标准/责任人，并标注无法自动验证项（真实 WPS 宿主）与责任人（用户）。
- [x] Git 策略含分支名、Draft PR 标题、PR 描述草稿、单 PR 决策依据与合并顺序（D-G）。
- [x] 所有歧义（O1–O8）进入待决事项，未擅自作出产品决策。
- [x] 本次仅写入 `plans/20260830-wps-support.md`（合规规划路径），未创建/覆盖任何源码、测试、配置、脚本或生成物。
- [x] Changelog 表记录 rev 1 + 2026-08-30 + 重梳说明 + 8 项访谈决策。
- [x] 状态标记为「待批准」，未自标已批准。
