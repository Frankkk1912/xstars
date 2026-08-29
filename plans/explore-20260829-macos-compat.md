# XSTARS macOS 平台兼容性调研报告

**主题**: XSTARS 拓展 macOS 平台支持  
**调研日期**: 2025-01  
**项目版本**: 1.1.1 (`pyproject.toml:4`)  
**WPS 参照分支**: `origin/feature/wps-support` (SHA `59c225a`)  
**报告路径**: `plans/explore-20260829-macos-compat.md`

---

## 1. Implementation Map

### 总体架构

XSTARS 是一个 Python 统计可视化引擎，通过 **xlwings** 桥接 Excel，以 `.xlsm` 工作簿 + 自定义 Ribbon 的形式交付。数据流如下：

```text
用户点击 Ribbon 按钮
    │
    ▼
VBA 回调 (ribbon_callbacks*.bas)
    │
    ├─── 开发者模式: RunPython "import xstars.main; xstars.main.run()"
    │         └── xlwings 运行时直接调用 Python
    │
    └─── 独立模式: Shell "xstars.exe run <workbook_path>"
              └── cli.py → xw.Book(workbook_path) → book.set_mock_caller()
                      └── main.py 各 run_*() 函数

main.py: run_*(book)
    │
    ├── book.selection → DataHandler.clean() → wide DataFrame
    ├── SettingsDialog(tkinter) → PrismConfig
    ├── StatsEngine(config).analyze(df) → StatsResult
    ├── PlotEngine(config).plot(df, result) → matplotlib Figure
    ├── sheet.pictures.add(fig, ...) → 插入 Excel
    └── book.app.status_bar = "..."
```

### 模块连接

| 模块 | 角色 | 平台耦合 |
| ------ | ------ | --------- |
| `ribbon/customUI14.xml` | Ribbon UI 定义 | Office 2010+ (Win+Mac) |
| `ribbon/ribbon_callbacks.bas` | VBA 回调 (开发者模式，RunPython) | Excel for Mac 支持 RunPython |
| `ribbon/ribbon_callbacks_installed.bas` | VBA 回调 (安装包，Shell + 注册表) | **Windows 专属** |
| `ribbon/ribbon_callbacks_standalone.bas` | VBA 回调 (便携，Shell + .exe) | **Windows 专属** |
| `xstars/main.py` | 主编排层 + 所有 run_*() 入口 | 部分 Windows 耦合 |
| `xstars/cli.py` | PyInstaller 冻结可执行文件入口 | **Windows 专属 (.exe)** |
| `xstars/config.py` | 配置持久化 (`~/.xstars/settings.json`) | 跨平台 ✅ |
| `xstars/data_handler.py` | Excel 选区读取 + pandas 清洗 | 跨平台 ✅ |
| `xstars/stats_engine.py` | 统计决策树 (scipy) | 跨平台 ✅ |
| `xstars/plot_engine.py` | matplotlib 图表生成 | 跨平台 ✅ |
| `xstars/styles.py` | matplotlib rcParams 主题系统 | 跨平台 ✅ |
| `xstars/ui_dialog.py` | tkinter/ttkbootstrap 设置对话框 | 跨平台 ✅ (有已知 macOS 小差异) |
| `xstars/presets/*.py` | 实验预设 (WB/qPCR/CCK8/ELISA) | 跨平台 ✅ |
| `xstars/tools/*.py` | 标准曲线工具 + 对话框 | 跨平台 ✅ |

---

## 2. Key Files and Symbols

### 平台耦合关键符号

| 文件 | 符号 / 机制 | 行范围 | 耦合性质 | 严重度 |
| ------ | ------------ | -------- | --------- | -------- |
| `xstars/main.py` | `_export_shape_highres()` | 897–942 | `PIL.ImageGrab.grabclipboard()` + `shape.CopyPicture(1,2)` (COM) | **BLOCKER** |
| `xstars/main.py` | `_get_selected_shapes()` | 881–894 | `book.app.api.Selection.ShapeRange.Item(i)` (COM API) | **HIGH** |
| `xstars/main.py` | `_select_sample_data()` | 1385–1413 | `book.app.api.InputBox(Type=8)` | MEDIUM |
| `xstars/main.py` | `_show_error()` | 166–199 | `root.attributes("-topmost", True)` (tkinter) | LOW |
| `xstars/main.py` | `run_reset_settings()` | 内含 `book.app.macro("MsgBox")` | 1456 附近 | LOW |
| `ribbon/ribbon_callbacks_installed.bas` | `ExePath()` | 全文 | `WScript.Shell.RegRead("HKCU\Software\XSTARS\InstallPath")` | **BLOCKER** (安装模式) |
| `ribbon/ribbon_callbacks_installed.bas` | `RunCmd()` | 全文 | `Shell "..." vbHide` + `\xstars\xstars.exe` | **BLOCKER** |
| `ribbon/ribbon_callbacks_standalone.bas` | `ExePath()` | 全文 | `ThisWorkbook.Path & "\xstars\xstars.exe"` 路径分隔符 `\` | **BLOCKER** |
| `xstars/cli.py` | `main()` | 全文 (24 行) | 仅假定 `.exe` 可执行文件命名，逻辑跨平台 | LOW |
| `pyproject.toml` | `[project]` | 9–22 | 无打包/分发脚本发现，无 PyInstaller spec 在版本库中 | — |

### 跨平台安全符号

| 文件 | 符号 | 行范围 | 说明 |
| ------ | ------ | -------- | ------ |
| `xstars/config.py` | `DEFAULT_SETTINGS_PATH` | 12 | `Path.home() / ".xstars" / "settings.json"` — 完全跨平台 |
| `xstars/config.py` | `PrismConfig.save/load` | 164–218 | JSON + pathlib，无 OS 依赖 |
| `xstars/data_handler.py` | `DataHandler.clean()` | 71–79 | 纯 pandas |
| `xstars/plot_engine.py` | `export_figure()` | 21–23 | `fig.savefig(path, ...)` — 跨平台 |
| `xstars/styles.py` | `BASE_RCPARAMS["font.sans-serif"]` | 26 | `["Arial", "Helvetica", "DejaVu Sans"]`；macOS 原生有 Arial/Helvetica |
| `tests/conftest.py` | 所有 fixtures | 全文 | 零 COM 依赖；使用 numpy/pandas |
| `tests/test_end_to_end.py` | `_xw_mocks()` + `TestPipelinePython` | 32–61 | 通过 `unittest.mock` 完全模拟 xlwings |

### xlwings 关键 API 调用

| 调用位置 | API | macOS 可用性 |
| --------- | ----- | ------------ |
| `main.py` 各 `run_*()` | `xw.Book.caller()` | ✅ (AppleScript 桥接) |
| `main.py` | `book.selection.sheet` / `sel.options()` | ✅ |
| `main.py` | `sheet.pictures.add(fig, ...)` | ✅ xlwings 原生支持 |
| `main.py` | `book.app.status_bar = "..."` | ✅ |
| `main.py:881` | `book.app.api.Selection.ShapeRange` | ⚠️ AppleScript 代理，行为可能不同 |
| `main.py:1393` | `book.app.api.InputBox(Type=8)` | ⚠️ Excel for Mac 支持但需验证 |
| `ribbon_callbacks.bas` | `RunPython "..."` | ✅ (需 `xlwings addin install`) |

---

## 3. Existing Patterns and Conventions

### 命名约定

- 公共入口函数：`run_*()`，所有位于 `xstars/main.py`，由 VBA 回调通过 `RunPython` 或 Shell 调用。
- 私有实现：`_run_*_impl()` 前缀，在 `run_*()` 中 try/catch 包裹。
- 错误处理模式（`main.py:201–208`）：统一通过 `_show_error(book, msg)` 路由，先尝试 tkinter 弹窗，失败则退回 `book.app.macro("MsgBox")`。

### 多后端分发约定（已有先例）

- `ribbon/` 目录中已存在三种 VBA 变体：`ribbon_callbacks.bas`（RunPython 开发模式）、`ribbon_callbacks_installed.bas`（安装包 Shell 模式）、`ribbon_callbacks_standalone.bas`（便携 Shell 模式）。这一**三文件分叉模式**是 macOS PR 可以效仿的参照。
- WPS 分支（SHA `59c225a`）建立的抽象机制尚未能直接 diff（工具限制），但预期其新增 WPS 专属 `.bas` 变体，延续同一模式。

### 配置持久化约定

- 所有用户配置经 `PrismConfig.to_dict()` 去掉瞬态字段后序列化为 JSON，存入 `~/.xstars/settings.json`（`config.py:12`）。路径通过 `pathlib.Path.home()` 构建，**完全跨平台**，macOS 无需改动。

### 测试约定

- `tests/conftest.py`：fixtures 全部为纯数据，零平台依赖。
- `tests/test_end_to_end.py`：通过 `unittest.mock.patch("xlwings.Book.caller", ...)` mock xlwings，不需要真实 Excel 进程。
- 所有测试均可在 macOS 上运行，**无需 Excel**。
- `matplotlib.use("Agg")` 在测试入口显式设置，避免 GUI 后端依赖。

### 打包/分发约定

- `pyproject.toml` 仅声明 `setuptools` 构建；没有 PyInstaller `.spec` 文件在版本库中（发布包构建脚本不在仓库内，属于证据缺口）。
- 独立分发当前依赖 PyInstaller `.exe`（Windows 专属）。

---

## 4. Candidate Implementation Paths

### 路径 A：xlwings 原生跨平台（开发者模式优先）

**依托扩展点**: `ribbon/ribbon_callbacks.bas`（RunPython 模式）+ xlwings macOS AppleScript 后端

**核心思路**:  
xlwings 自 v0.28 起在 macOS 通过 AppleScript 支持 Excel for Mac，`xw.Book.caller()`、`sheet.pictures.add(fig)`、`book.app.status_bar` 均可用。开发者模式下，VBA 通过 `RunPython` 调用 Python，这条路径在 macOS 上**理论上已经工作**，仅需解决两处阻塞：

1. **`_export_shape_highres()`**：将 `PIL.ImageGrab.grabclipboard()` + `shape.CopyPicture()` 替换为直接使用 `matplotlib` Figure 对象重新导出（图表原始 Figure 已有引用，`fig.savefig(path)` 可精确复现）。这是功能等价替换，无用户可见差异。
2. **`_get_selected_shapes()`**：`book.app.api.Selection.ShapeRange` 在 macOS AppleScript 代理上可能行为不同。需要 try/except 包裹并提供 fallback 到 `sheet.pictures`（当前已有 fallback 逻辑，`main.py:887–894`）。
3. **VBA 变体**：新增 `ribbon/ribbon_callbacks_mac.bas`（`RunPython` 模式，与现有 `ribbon_callbacks.bas` 内容基本相同，无需 `Shell` 调用）。

**主要改动面**:

- `xstars/main.py`: `_export_shape_highres()`（~46 行）、`_get_selected_shapes()`（~14 行）、`_show_error()` 中 `root.attributes("-topmost")` 的 macOS 兼容性（可选）
- `ribbon/ribbon_callbacks_mac.bas`：新建，复用 `ribbon_callbacks.bas` 内容
- `ribbon/README.md`：更新安装说明
- `requirements.txt`/`pyproject.toml`：无新依赖（Pillow 已为 `_export_shape_highres` 隐式依赖，macOS 上 `ImageGrab.grabclipboard()` 在 Pillow ≥9.1 可用，但整个函数可直接替换）

**代价与复杂度**: **低**。核心 Python 引擎已经跨平台，只需外科手术式修改 2 个函数 + 新建 VBA 文件。

**兼容性与回归影响**:

- Windows 路径不受影响（通过 `sys.platform` 或 try/except 分支保护）
- 现有测试全部可继续通过（测试不调用 `_export_shape_highres`）
- 仅支持"开发者模式"（需要 Python 环境 + `xlwings addin install`），不覆盖独立分发场景

**适用条件**: 实验室用户可以安装 Python 环境；macOS 独立分发作为后续 PR；快速验证可行性。

---

### 路径 B：macOS 独立分发（.app 束 + mac VBA Shell 替代方案）

**依托扩展点**: `xstars/cli.py` + 新建 `ribbon/ribbon_callbacks_mac_standalone.bas`

**核心思路**:  
将路径 A 的 VBA 调用从 `Shell "xstars.exe ..."` 替换为 macOS VBA 的等价机制。Excel for Mac 的 VBA 不支持 `Shell`，但支持 `MacScript`（AppleScript）调用外部命令，或通过 `Application.Run` 直接触发 Python（在 xlwings RunPython 模式下）。对于真正的独立分发，Python 应用可以打包为 `.app`（PyInstaller 支持 macOS 目标），通过 VBA 的 `MacScript` 来启动：

```vba
' macOS VBA 等价 Shell 调用
MacScript "do shell script ""/path/to/xstars run '" & wb & "'"""
```

或者采用 xlwings 独立模式（`xlwings standalone`），将 xlwings 运行时嵌入 `.xlsm` 中，通过 `RunPython` 而不是外部进程。

**主要改动面**:

- `ribbon/ribbon_callbacks_mac_standalone.bas`：新建，将 `Shell` 换为 `MacScript` + do shell script
- `xstars/cli.py`：几乎无需修改（逻辑跨平台，仅 argv 解析）
- PyInstaller 构建配置：需要新增 macOS `.spec` 文件（当前不在仓库中）
- `xstars/main.py`：同路径 A 的 `_export_shape_highres` 修复

**代价与复杂度**: **中**。`MacScript` 调用 shell 在 Excel for Mac VBA 有沙盒限制，路径处理复杂；PyInstaller macOS 包构建有额外复杂度（codesign、notarization）。

**兼容性与回归影响**:

- 不影响 Windows 路径
- macOS Excel 对 `MacScript` 的沙盒策略（Office 2019+ 有权限限制）需要实际验证
- 这是唯一能实现"macOS 零 Python 安装"目标的路径

**适用条件**: 目标用户群无 Python 环境；需要与 Windows 安装包体验对等；长期维护目标。

---

### 路径 C：仅 RunPython 模式 + Excel for Mac 官方 add-in 机制

**依托扩展点**: `customUI14.xml` + xlwings 官方 macOS 安装指南

**核心思路**:  
不追求独立分发，仅提供开发者/高级用户模式：用户通过 `pip install xstars && xlwings addin install` 安装，然后打开 `XSTARS_Templates.xlsx`（已有模板文件在仓库根），按照新的 macOS 安装说明导入 `ribbon_callbacks_mac.bas`。产品上这是最保守但最快速的 MVP。  

相比路径 A，差异在于**显式放弃**独立分发目标，文档层面声明支持范围：macOS + Excel for Mac + Python 3.10+。

**主要改动面**:

- `xstars/main.py`：同路径 A（`_export_shape_highres` 修复为核心）
- `ribbon/ribbon_callbacks_mac.bas`：新建（等同 `ribbon_callbacks.bas`，或甚至直接复用）
- `ribbon/README.md`：新增 macOS 安装小节
- `README.md`：更新平台要求

**代价与复杂度**: **最低**。变更范围极小，回归风险极低。

**兼容性与回归影响**: 无回归风险；Windows 路径零改动。

**适用条件**: 首个 macOS PR 的 MVP 目标；后续可用路径 B 补充独立分发。

---

## 5. Preliminary Gap Analysis

| 缺口 | 现状证据 | 影响范围 | 严重度 |
| ------ | --------- | --------- | -------- |
| **Export 功能**: `_export_shape_highres()` 使用 `PIL.ImageGrab.grabclipboard()` + COM `shape.CopyPicture()` | `main.py:897-942` — 依赖 Windows clipboard API 和 COM Shape 对象 | "Export Image" 按钮在 macOS 完全不可用 | **BLOCKER** |
| **VBA 独立模式**: `Shell "...\xstars.exe ..."` | `ribbon_callbacks_installed.bas` + `ribbon_callbacks_standalone.bas` — 硬编码 `\` 路径分隔符和 `.exe` | 独立/安装包模式在 macOS 完全无效 | BLOCKER（独立模式） |
| **注册表路径**: `WScript.Shell.RegRead("HKCU\Software\XSTARS\InstallPath")` | `ribbon_callbacks_installed.bas:13` | 安装包模式仅限 Windows，macOS 无注册表 | BLOCKER（安装包模式） |
| **`book.app.api` 子调用兼容性**: `Selection.ShapeRange` / `InputBox(Type=8)` | `main.py:882–884`, `main.py:1392` | 在 macOS xlwings AppleScript 代理下行为未经验证；`InputBox(Type=8)` 在 Excel for Mac 有限制 | HIGH |
| **tkinter `-topmost` 属性**: `root.attributes("-topmost", True)` | `main.py:182` — macOS tkinter 下 `-topmost` 行为与 Windows 不同 | 错误弹窗可能被 Excel 窗口遮挡 | LOW |
| **字体渲染**: `Arial` 在 macOS 系统字体中存在，但 matplotlib 字体探测路径不同 | `styles.py:26` | 图表字体可能回退到 DejaVu Sans（matplotlib bundled），视觉细节差异 | LOW |
| **xlwings addin install 用户引导**: 安装流程与 Windows `.exe` 完全不同 | README.md 当前仅描述 Windows 安装包 | 用户体验缺口，非代码问题 | MEDIUM |
| **PyInstaller macOS 构建**: 无 `.spec` 文件在仓库 | `pyproject.toml` 无 PyInstaller 配置 | 独立分发需要额外构建基础设施 | HIGH（独立模式） |

---

## 6. Open Questions

1. **`InputBox(Type=8)` 在 Excel for Mac 的行为**：xlwings 文档未明确说明 `book.app.api.InputBox(Type=8)`（用于鼠标选区的 InputBox）在 macOS AppleScript 代理下是否正常工作。如果不支持，ELISA 和标准曲线的样本选择流程需要替代 UI（如 tkinter 对话框提示用户手动输入范围地址）。**需要在 Excel for Mac 实际验证**，或查阅 xlwings macOS changelog。

2. **`Export Image` 功能在 macOS 的替代策略**：`_export_shape_highres()` 通过 COM CopyPicture 获取 Shape 的光栅图像。在 macOS 上，由于图表原本就是 matplotlib Figure 对象（`sheet.pictures.add(fig, ...)` 的 `fig`），理论上可以通过重新渲染导出。但 macOS PR 是否需要保留对 **非 XSTARS 生成的** Excel 形状的导出能力（即用户选中任意 Excel 图表然后 Export）？如果需要，macOS 路径更复杂（需要 AppleScript 截图方案）。

3. **macOS 目标用户群的 Python 安装情况**：是否要求用户预先安装 Python（开发者/高级用户模式），还是必须实现"零 Python 安装"（独立分发）？这决定了路径 A/C（低复杂度）vs 路径 B（高复杂度）的优先级。

4. **WPS for Mac 的关系**：WPS Office 存在 macOS 版本。本 macOS PR 是否需要同时考虑 WPS for Mac，还是严格只针对 Microsoft Excel for Mac？

5. **最低 macOS 版本支持**：xlwings macOS 对 Excel for Mac 2016+ 有最佳支持，Excel 2011 for Mac 不支持 xlwings。目标 macOS/Office 版本范围？

6. **独立分发渠道**：macOS 独立分发应交付为 `.app` bundle（PyInstaller）、通过 Homebrew tap，还是仅支持 `pip install`？影响路径 B 的构建基础设施工作量。

7. **证据缺口**：WPS 分支（SHA `59c225a`）的具体文件变更无法在只读环境中通过 git diff 直接获取（无 bash 工具）。WPS PR 中多平台/多 Office 套件的抽象层设计需要用户提供 diff 摘要，或 cartographer 在有 bash 工具的环境中补充分析。

---

## 6b. WPS PR（origin/feature/wps-support）差异分析（补充，由主编排器完成）

原 Open Question 7 已由主编排器通过 `git diff main...origin/feature/wps-support --stat` 补齐（SHA `59c225a`）：

**WPS PR 文件清单（21 文件，+5228 行）**：

- `docs/cross-platform-office-technology-strategy.md`（114 行）：**已确定的项目级跨平台技术路线**
- `docs/wps-support-implementation-plan.md`（510 行）：WPS 支持实施 Plan
- `poc/wps/addin/`：WPS JS 加载项（manifest.xml、ribbon.xml、ribbon.js、index.html、vite 构建、离线发布脚本）
- `poc/wps/probe_server.py`（280 行）+ `poc/wps/service_server.py`（424 行）：本地 Python HTTP 服务桥
- `tests/test_wps_probe.py`（139 行）：probe 测试

**关键架构决策（已确认，摘自 strategy 文档）**：

- 统一 Python 核心与通信契约，按宿主提供独立适配器；Python Core 不直接依赖宿主 API
- **Excel macOS 推荐路线 = VBA + xlwings（保留现有实现 + Mac 适配）**，状态为"后续增加 Mac 适配"
- Excel 迁移 Office.js 被明确否决（离线部署复杂度 > 收益）
- WPS macOS 暂不承诺正式支持，需先 PoC
- macOS 需单独适配：安装、路径、AppleScript/App Sandbox、签名公证、图片导出
- Windows 与 macOS 使用独立安装包和签名流程
- xlwings macOS 不支持 UDF，但 XSTARS 不依赖 UDF，非障碍

**对 macOS PR 的直接影响**：本 PR 应作为 strategy 文档中 "Excel macOS: VBA + xlwings" 路线的落地实施，且目标架构图中 "Direct Python Bridge → macOS adapter" 即为本 PR 的交付物。

## 7. Persisted Report

本报告已写入：`/Users/frank/Documents/GitHub/xstars/plans/explore-20260829-macos-compat.md`

> **Project memory notice**: 本次调研同时将架构摘要写入了  
> `/Users/frank/.pi/agent-memory/codebase-cartographer/MEMORY.md`  
> 建议将 `.pi/agent-memory/` 加入项目 `.gitignore`（除非明确决定版本化这些记录）。
