# Plan: 修复 Windows PyInstaller 冻结包 pandas `_pandas_datetime_CAPI` 报错与安装器重打包

- **状态：待审批**（待用户批准后进入实施；D1 版本号裁定为实施前置）
- **日期：2026-10-07**
- **rev：1**（2026-10-07 初稿）
- **目标基线：`main` @ `b5705915`**（2026-10-07 实测 HEAD；module 缓存构建于 `d654c932`，以 `b5705915` 为分支起点）
- **PR 目标（工作）分支：`fix/windows-installer-pandas-capi`**（任务书显式指定，覆盖默认 `feat/<slug>` 命名约定；PR base = `main`）
- **输入**：任务书《【任务目标】撰写正式实施方案文档》（2026-10-07，含问题定位与根因结论，下称"问题简报"）+ 本仓库代码实读取证（xstars.spec / xstars/cli.py / installer/excel/* / installer/wps/xstars-wps.spec / tests/test_macos_installer.py）

| Changelog | 日期 | 变更摘要 | 依据 |
| --- | --- | --- | --- |
| rev 1 | 2026-10-07 | 初稿：9 段结构，M1–M2 共 2 个 Milestone、8 个 To-do 任务、6 个功能缺口；含锚点实测校正（xstars.spec pandas 硬编码条目实测 :78-81，问题简报引 :76-80）与新发现的第三处版本源 `installer/excel/XSTARS.iss:13` | 问题简报 + 代码实读（见 §4 证据锚点） |

> 锚点标注约定：本文行号均为 feature-planner 于 2026-10-07 在 `main` @ `b5705915` 上实读核验（read 工具逐行计数）。问题简报引用的 `xstars.spec:76-80` 与实测 `:78-81`（含注释行 :78）存在约 2 行漂移，**以本文实测锚点为准**，语义结论（仅硬编码 3 个 `pandas._libs.tslibs` 模块）不受影响。

---

## 1. Goal

修复 Windows 版 XSTARS v1.2 安装后在 Excel 中点击任意选项即崩溃（`Failed to execute script 'cli' due to unhandled exception: partially initialized module 'pandas' has no attribute '_pandas_datetime_CAPI'`）的打包缺陷，并交付可重复的 Windows 重打包与验收流程：

1. **终端用户价值（可验证结果）**：Windows 用户安装重打包后的 `XSTARS_Setup_v*.exe` 后，在 Excel 点击 XSTARS 功能区任意按钮（Quick Run / Run / Export 等）正常执行统计分析与导出，**不再出现任何崩溃对话框**；该修复由开发侧重打安装包解决，用户无需改动本机 Office/系统环境。
2. **构建质量价值**：`xstars.spec` 全量收集 pandas 的数据文件、C 扩展动态库与子模块（`collect_all("pandas")`），使 pandas 2.x C-API（`pandas_datetime`/`pandas_parser` 等）在冻结包内完整初始化；`xstars/cli.py` 增加防御性 pandas 预热，使失败点前移、可诊断。
3. **流程价值**：交付 Windows 开发机清晰的构建与烟测验证 Runbook（§8 + M2）——无需打开 Excel 即可完成 headless 冒烟判据，Excel 实机验证为最终验收。
4. **版本价值**：以补丁版本（建议 1.2.1，见待决 D1）重新发行 Windows 安装包，三处版本源（`xstars/__init__.py` / `pyproject.toml` / `installer/excel/XSTARS.iss`）同步一致。

## 2. Requirements

| ID | 需求 | 来源 | 约束级别 |
| --- | --- | --- | --- |
| R1 | **`xstars.spec` 全量收集 pandas**：引入 `PyInstaller.utils.hooks.collect_all("pandas")`，将其返回的 `datas`、`binaries`、`hiddenimports` 三元组分别全量注入 `Analysis(...)` 的 `datas=`、`binaries=`、`hiddenimports=` 参数（现有 3 条 `pandas._libs.tslibs.*` 硬编码条目保留，冗余无害、最小 diff） | 问题简报 R1；根因结论（xstars.spec:78-81 仅硬编码 3 个模块 + binaries=[]） | 〔硬〕 |
| R2 | **`xstars/cli.py` 防御性预热 pandas**：在模块级 import 区（xstars/cli.py:13-14 之后）显式 `import pandas`（`# noqa: F401` + 注释说明用途），保证任何调用路径（含无 Excel 的 CLI 冒烟）都在 `import_module("xlwings")`（xstars/cli.py:54）之前完成 pandas C-API 初始化 | 问题简报 R2 | 〔硬〕 |
| R3 | **版本号策略**：升级为 **1.2.1** 补丁版本（建议值，最终以 D1 裁定为准），三处同步：`xstars/__init__.py:3`（`__version__`）、`pyproject.toml:7`（`version`）、`installer/excel/XSTARS.iss:13`（`MyAppVersion`，另 :2 头注释）。前两处由既有防漂移测试 `tests/test_macos_installer.py::test_package_version_matches_pyproject`（:66-69）强制一致；`XSTARS.iss` 由安装器产物名验证兜底（`build_installer.py:253-260` 按 `_extract_version()` 期望 `XSTARS_Setup_v{version}.exe`） | 问题简报 R3；代码实读（新发现第三处版本源 XSTARS.iss:13） | 〔硬〕 |
| R4 | **Windows 开发机本地测试与烟测命令规范**：提供无需打开 Excel 的 headless 冒烟（冻结包 `--help`/无参数调用优雅退出 + pandas 二进制清单核查）与构建前/后命令序列；判据可逐条打勾 | 问题简报 R4 | 〔硬〕 |
| R5 | **Inno Setup 重新编译与产物验证**：经 `installer/excel/build_installer.py` 第三步（ISCC 编译 `installer/excel/XSTARS.iss`）产出 `installer/excel/output/XSTARS_Setup_v1.2.1.exe`，验证存在性、文件名版本、大小与 SHA256 | 问题简报 R5 | 〔硬〕 |
| R6 | **平台隔离不破坏**：macOS pkg 链路（`installer/mac/`，非 PyInstaller）与 WPS 链路（`installer/wps/`，独立 spec 且 `excludes=["xlwings"]`）零改动；核心统计分析业务算法零改动 | 问题简报非目标项 + plans/20260906-macos-installer.md R17 | 〔硬〕 |
| R7 | **验收标准**：Windows 开发机烟测（headless）+ Excel 实机点击验证通过，且仓库全量 pytest 无新增失败，作为合并前置 | 问题简报背景（必须由开发侧重打解决）+ 仓库既有门禁惯例 | 〔硬〕 |

## 3. Non-goals

本次明确**不做**：

1. **不修改 macOS pkg 逻辑**：`installer/mac/**`（build_pkg.py、postinstall.sh、uninstall.sh、distribution.xml 等）与 `docs/macos-*` 零改动。macOS 发布链路使用 python-build-standalone 运行时 + 真实 site-packages（plans/20260906-macos-installer.md R17「不使用 PyInstaller」），不存在冻结包漏收集问题。
2. **不改动 WPS 逻辑**：`installer/wps/**`（含 `xstars-wps.spec`、`XSTARS_WPS.iss`、`wps_helper.py`）、`wps-addon/**`、`xstars/wps_service.py` 零改动。WPS 使用独立 spec 且 `excludes=["xlwings"]`（installer/wps/xstars-wps.spec:51，实测），崩溃链路前置条件（xlwings 导入）不存在，问题简报认定其不受影响；本次不验证、不修复（残余风险见 §9.2 RK-7）。
3. **不改动核心统计分析业务算法**：`xstars/main.py`、`xstars/stats_engine.py`、`xstars/plot_engine.py`、`xstars/presets/**`、`xstars/application/**` 零改动。
4. **不改依赖版本**：不降级/升级 pandas、numpy 等任何依赖（`pyproject.toml:12-22` 依赖声明不动）；不改 PyInstaller 打包模式（保持 one-dir）。
5. **不修改 `installer/excel/build_installer.py`**：既有三步构建流程（清理 :67-77、`--clean` :79-82、ISCC :251、产物名校验 :253-260）已满足 R4/R5，无需改动。
6. **不新增自动化测试文件/用例**：本次改动面为打包配置与 2 行源码防御，冻结产物行为无法由 pytest 覆盖；验证由 §8 的命令行/人工验证承担（V3、V8–V12）。如需 spec 文本断言测试，记入残余项另议。
7. **不做代码签名/SmartScreen 处理**、不引入 CI 出 Windows 安装包（本次仍为 Windows 开发机本地出包；CI 仅跑既有 windows-tests 回归）。
8. **不改 `ribbon/*.bas`、VBA、XSTARS.xlam 制作逻辑**（`build_installer.py` Step 2 原样复用）。

## 4. Research summary

### 4.1 故障与根因（问题简报结论 + 代码实证）

- **故障现象**：Windows 用户安装 XSTARS v1.2 后，Excel 点击任意选项报 `Failed to execute script 'cli' due to unhandled exception: partially initialized module 'pandas' has no attribute '_pandas_datetime_CAPI' (most likely due to a circular import)`（问题简报）。
- **崩溃链路**：`cli.py:68`（`main()` 调用，实测为 `if __name__ == "__main__"` 下的 `main()`）→ `cli.py:54`（`xw = import_module("xlwings")`，实测锚点一致）→ xlwings conversion 层 → pandas → `pandas/_libs/missing.pyx:40` 访问 `_pandas_datetime_CAPI` 失败（问题简报堆栈）。
- **用户机无关性（实证支持）**：崩溃发生在 `import_module("xlwings")`（xstars/cli.py:54）**之前即失败于 pandas 初始化**，而读取工作簿的代码 `book = xw.Book(workbook_path)`（xstars/cli.py:56）根本未执行到——与用户系统/Office 环境无关，必须由开发侧重打包解决（问题简报结论，与实测锚点吻合）。
- **根因（打包配置缺陷，实测）**：
  - `xstars.spec:53` `binaries=[]`——零二进制显式收集；
  - `xstars.spec:54-61` `datas=` 仅含 matplotlib mpl-data、ttkbootstrap、statannotations 三项，**无 pandas 数据文件**；
  - `xstars.spec:78-81`（注释行 :78）"pandas internals" 仅硬编码 3 个子模块：`pandas._libs.tslibs.timedeltas`、`pandas._libs.tslibs.np_datetime`、`pandas._libs.tslibs.nattype`——**未收集 pandas 2.x 的 C 扩展动态库（.pyd，如 `pandas_datetime`/`pandas_parser` 及 `pandas/_libs`、`pandas/_libs/tslibs` 下全部扩展）与包数据**，导致 Windows 冻结包内 pandas C-API 初始化失败（问题简报根因结论，与 spec 实测一致）。
  - 任务简报引 "xstars.spec 第 76-80 行"，实测锚点为 :78-81（含注释行），语义结论不变（锚点校正见文首约定）。

### 4.2 平台隔离性（实证支持，佐证 Non-goals）

- **macOS**：发布链路 `installer/mac/build_pkg.py` 不使用 PyInstaller（plans/20260906-macos-installer.md R17：xlwings `RunFrozenPython` 仅 Windows 可用，`RunPython` 需真实解释器）；运行时为 python-build-standalone + staging 真实 `pip install` 的 site-packages（同 Plan rev 1/rev 7），非冻结运行时，**不可能**出现 PyInstaller 漏收集问题。
- **WPS**：`installer/wps/xstars-wps.spec:51` `excludes=["xlwings"]`（实测），并采用 `collect_data_files`/`collect_submodules`（installer/wps/xstars-wps.spec:17-18，实测）；问题简报认定 WPS 不受此问题影响（崩溃链路前置条件 `import_module("xlwings")` 不存在，WPS 走 `serve`/`worker` 模式，xstars/cli.py:42-45）。

### 4.3 现有模式（推荐方案的仓库内先例）

- **PyInstaller hooks 工具已有使用先例**：`installer/wps/xstars-wps.spec:17-18` 已使用 `from PyInstaller.utils.hooks import collect_data_files, collect_submodules`。`collect_all` 与之同族，为 PyInstaller 官方 `utils.hooks` 公开 API，语义为"返回 `(datas, binaries, hiddenimports)` 三元组，全量收集指定包的数据文件、二进制与子模块"——正对本根因（缺 .pyd 与数据文件）。
- **版本同步已有强制机制**：`tests/test_macos_installer.py::test_package_version_matches_pyproject`（:66-69）断言 `xstars.__version__` 与 `pyproject.toml` 版本一致；构建脚本读版本源：`xstars.spec:23-24`（APP_VERSION 正则抓 `xstars/__init__.py`）、`installer/excel/build_installer.py:46-52`（`_extract_version()` 同源）。**新发现**：`installer/excel/XSTARS.iss:13` 另有硬编码 `#define MyAppVersion "1.2.0"`（:2 头注释亦写死 `XSTARS_Setup_v1.2.0.exe`），是第三个版本源，且无自动校验——若不随动，`build_installer.py:253-260` 将只打 WARNING 而产物名仍为 v1.2.0，版本校验失效。故 R3 范围为**三处**而非任务书所述两处。
- **Windows 构建流程现状**：`installer/excel/build_installer.py` 三步链——① PyInstaller（`--clean` + 预清理 `build/`、`dist/`，:67-82）→ ② Excel COM 出 `XSTARS.xlam` → ③ ISCC 编译 `XSTARS.iss` 并校验产物名（:248-260）。Runbook 直接复用，无需新脚本。

### 4.4 备选方案与取舍

| 方案 | 内容 | 取舍 | 结论 |
| --- | --- | --- | --- |
| **A（推荐）** | `collect_all("pandas")` 注入 datas/binaries/hiddenimports（R1）+ cli.py 预热（R2） | 一行 API 全量收集，覆盖 .pyd/数据/子模块；体积 +约 15-30MB（问题简报估计值） | **采用** |
| B | 手工逐个列全 `pandas._libs.*` hiddenimports | 随 pandas 版本漂移、清单脆弱；本次根因即"手工清单不全" | 否决 |
| C | `collect_data_files` + `collect_dynamic_libs` + `collect_submodules` 三组合 | 与 A 等价但三行手动组合，遗漏面同 B | 否决（A 的等价繁琐版） |
| D | 降级 pandas 1.x 规避 C-API | 与 `pyproject.toml:13` `pandas>=2.0` 冲突，改依赖面大、波及统计行为 | 否决 |
| E | 预热改用 PyInstaller runtime_hooks | 多一个构建产物、可测性差 | 否决（R2 采用显式 import，可读可测） |

### 4.5 外部参考

- PyInstaller 官方文档 `PyInstaller.utils.hooks.collect_all`（公开 API，返回 `(datas, binaries, hiddenimports)`；`collect_all` 内部聚合 `collect_data_files`/`collect_dynamic_libs`/`collect_submodules` 族）。此为 API 语义陈述，实施时以 Windows venv 实跑（§8 V3）确认返回值非空为准。

## 5. Gap analysis

| ID | 功能缺口 | 现状 | 影响 | 补齐任务 |
| --- | --- | --- | --- | --- |
| G1 | xstars.spec 未收集 pandas 的 C 扩展动态库与数据文件 | `binaries=[]`（xstars.spec:53）；`datas=` 无 pandas（:54-61）；仅 3 条 pandas tslibs 硬编码 hiddenimports（:78-81） | Windows 冻结包 pandas C-API 初始化失败，Excel 点击任意按钮即崩溃（v1.2 线上故障） | T1.1 |
| G2 | 冻结包内 pandas 初始化时机不受控，失败点深埋于 xlwings 链路 | `xstars/cli.py` 无 pandas 预热，首次触碰 pandas 在 `import_module("xlwings")`（:54）之后的第三方链路里 | 崩溃堆栈指向 missing.pyx，难诊断；冒烟无法区分打包缺陷与业务缺陷 | T1.2 |
| G3 | 版本三源不同步风险（补丁版本发行需同步升级） | `xstars/__init__.py:3` 与 `pyproject.toml:7` 为 1.2.0（有防漂移测试）；`installer/excel/XSTARS.iss:13` 独立硬编码 1.2.0（无自动校验） | 产物名/安装器版本与包版本漂移；`build_installer.py:260` 仅 WARNING 不 fail | T1.3 |
| G4 | 缺少无需 Excel 的冻结包 headless 烟测规程 | 现行触发路径只有 Excel→VBA→`xstars.exe run_quick <wbk>`（xstars/cli.py 文档字符串），无独立冒烟判据 | 修复有效性只能靠 Excel 实机试，验收慢且不可重复 | T2.2 |
| G5 | Windows 重打包 + Inno 产物验证流程未成文、无产物级判据 | `build_installer.py` 出包后仅打印大小，无哈希/清单核查 | 构建环境缓存污染或漏收集无法被发现（本次事故的复发路径） | T2.1、T2.3 |
| G6 | Excel 实机最终验收无归属与记录位置 | 上游报告由用户人工发现，无验收记录归档 | 修复闭环缺失 | T2.4 |

**自查：无孤立缺口**——G1→T1.1；G2→T1.2；G3→T1.3；G4→T2.2；G5→T2.1+T2.3；G6→T2.4。反向核查：8 个任务中 T1.4 为 M1 验证汇总（支撑 G1–G3 的验收门），其余任务均与缺口一一/一对多对应，无范围外任务。

## 6. Milestone 表格

| Milestone | Status | Dependencies | Validation | Notes |
| --- | --- | --- | --- | --- |
| M1 规范与代码修改（xstars.spec / xstars/cli.py / 版本三处同步） | [ ] | 无（D1 版本号裁定为 T1.3 前置） | §8 V1–V7 全部通过 | 纯仓库改动，可在 macOS 开发机 + CI 完成 |
| M2 Windows 构建与验收 Runbook（清理重建 → headless 烟测 → Inno 产物验证 → Excel 实机） | [ ] | M1 | §8 V8–V12 全部通过（责任人=用户，Windows 开发机） | 全部执行动作在 Windows 开发机；结果回传 PR 评论 |

（Milestone 总数 2 ≤ 10，目标 ≤7 达成。）

## 7. 分 milestone 的 To-do checkbox 清单

### M1 规范与代码修改

- [ ] T1.1 `xstars.spec` 注入 `collect_all("pandas")`（R1 / G1）
  - 文件：`xstars.spec`（修改）
  - 修改：① 顶部 import 区（:14-16 附近）新增 `from PyInstaller.utils.hooks import collect_all`（参照 installer/wps/xstars-wps.spec:17-18 既有 hooks 用法），并执行 `pandas_datas, pandas_binaries, pandas_hiddenimports = collect_all("pandas")`（放在 `a = Analysis(...)` 之前）；② `Analysis(...)` 的 `binaries=[]`（:53）改为 `binaries=[*pandas_binaries]`；③ `datas=`（:54-61）列表尾部追加 `*pandas_datas`；④ `hiddenimports=`（:62 起）列表追加 `*pandas_hiddenimports`（保留既有 3 条 `pandas._libs.tslibs.*` 条目，:79-81，冗余无害、维持最小 diff）。不改 `excludes`、UPX、BUNDLE、COLLECT 等其余任何部分。
  - 验收：`python -m py_compile xstars.spec` 退出码 0；文本核查 `collect_all("pandas")` 与三处展开变量（`*pandas_datas`/`*pandas_binaries`/`*pandas_hiddenimports`）全部命中；Windows venv 实跑 `python -c "from PyInstaller.utils.hooks import collect_all; d,b,h=collect_all('pandas'); assert d and b and h"` 通过（§8 V3）。
  - 依赖：无

- [ ] T1.2 `xstars/cli.py` 防御性预热 pandas（R2 / G2）
  - 文件：`xstars/cli.py`（修改）
  - 修改：在模块级 import 区（`import sys` / `from importlib import import_module`，:13-14 之后、`def _run_serve_mode` 之前）新增 `import pandas  # noqa: F401  # defensive prewarm: initialize pandas C-API before xlwings lazy-imports pandas`。**不动** `main()`、`_run_serve_mode`、`_run_worker_mode` 任何逻辑（:17-64）；确保预热发生在 `import_module("xlwings")`（:54）可达之前。
  - 验收：`python -c "import xstars.cli"` 无异常（headless 可导）；`ruff check xstars/cli.py` 0 error（`# noqa: F401` 覆盖未使用导入）；pytest 全量无新增失败。
  - 依赖：无

- [ ] T1.3 版本同步升级至 1.2.1（R3 / G3；版本值以 D1 裁定为准）
  - 文件：`xstars/__init__.py`（修改 :3）、`pyproject.toml`（修改 :7）、`installer/excel/XSTARS.iss`（修改 :13，另 :2 头注释）
  - 修改：`__version__ = "1.2.1"`；`version = "1.2.1"`；`#define MyAppVersion "1.2.1"`；XSTARS.iss 头注释 `; Produces: XSTARS_Setup_v1.2.1.exe`。三处严格一致（若 D1 裁定其他版本号则按裁定值三处同改）。
  - 验收：三处 grep 均为新版本值；`python -m pytest tests/test_macos_installer.py::test_package_version_matches_pyproject` 通过（覆盖 __init__/pyproject 两处）；XSTARS.iss 版本一致性由 T2.3 产物名（`XSTARS_Setup_v1.2.1.exe`）兜底；全量 pytest 无新增失败。
  - 依赖：无（版本最终值待 D1 裁定，裁定前不得合入）

- [ ] T1.4 M1 静态核查与回归验证汇总
  - 文件：无（验证动作；发现问题回对应任务修复后重跑）
  - 修改：执行 §8 V1–V7 全部可执行项并逐项记录结果（本环境为 macOS，V3 需 Windows venv，可后置并入 M2 一并执行并标注）。
  - 验收：V1–V7 全部通过或标注明确的后置责任与原因。
  - 依赖：T1.1、T1.2、T1.3

### M2 Windows 构建与验收 Runbook（用户在 Windows 开发机执行）

- [ ] T2.1 Runbook W1–W2：构建前清理与全量重建（R4 / G5）
  - 文件：无代码修改（执行动作）；产物 `dist/xstars/`、`installer/excel/XSTARS.xlam`、`installer/excel/output/`
  - 修改：W1 环境准备——检出 `fix/windows-installer-pandas-capi`，`.venv` 内 `pip install pyinstaller xlwings`（build_installer.py 文档字符串 Prerequisites），安装 Inno Setup 6，确保**真实 Microsoft Excel**（非 WPS 劫持 CLSID）且已启用"Trust access to the VBA project object model"；记录 `pip show pandas pyinstaller` 版本入 PR 评论。W2 清理重建——删除 `build/`、`dist/`、`installer/excel/output/` 旧产物（`build_installer.py:67-77` 亦自动 rmtree `build/`、`dist/`；`:79-82` 传 `--clean` 清 PyInstaller 缓存，规避构建缓存污染 RK-2），随后运行 `.venv\Scripts\python.exe installer\excel\build_installer.py`（三步全量：PyInstaller → xlam → Inno）。
  - 验收：三步全绿；`dist\xstars\xstars.exe` 与 `installer\excel\output\XSTARS_Setup_v1.2.1.exe` 均生成；构建环境版本记录已回传。
  - 依赖：M1

- [ ] T2.2 Runbook W3：无 Excel 的 headless 烟测（R4 / G4）
  - 文件：无（执行动作）
  - 修改：① pandas 二进制清单核查——PowerShell `Get-ChildItem -Recurse dist\xstars -Include pandas_*.pyd`（PyInstaller ≥6.x one-dir 布局位于 `dist\xstars\_internal\pandas\...`，旧版位于 `dist\xstars\pandas\...`），确认至少包含 `pandas_datetime*.pyd`、`pandas_parser*.pyd` 与 `pandas\_libs\tslibs\*.pyd`（np_datetime/nattype/timedeltas 等）；② headless 冒烟——`cmd` 中 `start /wait "" dist\xstars\xstars.exe --help` 后 `echo %ERRORLEVEL%`（或无参数运行一次）。
  - 验收：清单含上述 .pyd 全集；冒烟进程**优雅退出**（exit code 1，`SystemExit("Usage: ...")` 语义，xstars/cli.py:47-48）且**无** `Failed to execute script 'cli' due to unhandled exception` 对话框——因 T1.2 预热使该路径即完成 pandas C-API 初始化，此为"根治判据"（若仍复现 → RK-5 升级调查，不得进入 T2.3 发布）。
  - 依赖：T2.1

- [ ] T2.3 Runbook W3：Inno Setup 重编译与产物验证（R5 / G5）
  - 文件：无（产物 `installer/excel/output/XSTARS_Setup_v1.2.1.exe`）
  - 修改：复核 `build_installer.py` 第三步输出（:248-260）：ISCC 编译 `XSTARS.iss`、按 `_extract_version()`（:46-52）期望 `XSTARS_Setup_v{version}.exe`；记录产物大小与 SHA256（`certutil -hashfile XSTARS_Setup_v1.2.1.exe SHA256`）入 PR 评论。
  - 验收：`installer\excel\output\XSTARS_Setup_v1.2.1.exe` 存在、文件名版本为 v1.2.1、大小与 SHA256 已记录；构建日志**无** `WARNING: Expected ... not found`（出现即视为 XSTARS.iss 版本漂移，回 T1.3 修复后重跑）。
  - 依赖：T2.1

- [ ] T2.4 Runbook W4：Excel 实机点击验证与结果回传（R7 / G6）
  - 文件：无（人工验证；责任人=用户，需真实 Excel 宿主，本环境无法执行）
  - 修改：安装 `XSTARS_Setup_v1.2.1.exe`，打开含 XSTARS 功能区的工作簿，逐一点击 Quick Run / Run / Export High-Res 等按钮（覆盖统计分析与导出两类核心操作）；确认无任何崩溃对话框、分析结果与图表导出正常；将验证结果（按钮清单逐项、截图、产物哈希）回传至 PR 评论归档。
  - 验收：全部按钮正常工作，无 `_pandas_datetime_CAPI` 报错；验收记录已归档 PR。
  - 依赖：T2.2、T2.3

## 8. Validation contract

| # | 检查项 | 命令/验证方式 | 预期结果 | 通过标准 | 责任人 |
| --- | --- | --- | --- | --- | --- |
| V1 | spec 语法有效 | `python -m py_compile xstars.spec` | 无输出 | 退出码 0 | Agent |
| V2 | spec 注入核查（R1） | 文本核查 `xstars.spec`：`from PyInstaller.utils.hooks import collect_all`、`collect_all("pandas")`、`[*pandas_binaries]`、`*pandas_datas`、`*pandas_hiddenimports` | 5 处全部命中；既有 3 条 tslibs 条目保留 | 全命中 | Agent |
| V3 | `collect_all("pandas")` 可用性（Windows venv） | `python -c "from PyInstaller.utils.hooks import collect_all; d,b,h=collect_all('pandas'); assert d and b and h; print(len(d),len(b),len(h))"` | 三个计数均 >0，`b` 含 pandas_datetime/pandas_parser 相关 .pyd | assert 通过 | 用户（Windows）；可后置并入 M2 |
| V4 | cli 预热（R2） | `python -c "import xstars.cli"`；`ruff check xstars/cli.py` | 导入无异常；0 error | 通过 | Agent |
| V5 | 版本三源一致（R3） | `python -m pytest tests/test_macos_installer.py::test_package_version_matches_pyproject -v`；核查 `XSTARS.iss:13` `MyAppVersion` 与 `xstars/__init__.py:3`、`pyproject.toml:7` 同值 | 测试通过；三处同为 1.2.1（或 D1 裁定值） | 通过 | Agent |
| V6 | 全量回归（R7） | `python -m pytest tests -v`（macOS 开发机 / CI windows-tests） | 无新增失败（历史基线参考：Windows 3.10 ≈ 410 passed/10 skipped，plans/20260906-macos-installer.md rev 17） | 相对基线 0 新增 failed | Agent + CI |
| V7 | CI 门禁 | PR 触发既有 workflow（含 `windows-tests`、macOS jobs、VBA 门禁） | 全绿 | 全绿 | CI |
| V8 | pandas 二进制收集核查（构建后，根治判据一） | PowerShell `Get-ChildItem -Recurse dist\xstars -Include pandas_*.pyd`（另查 `pandas\_libs\tslibs\*.pyd`） | `pandas_datetime*.pyd`、`pandas_parser*.pyd`、tslibs 扩展全部在位 | 全部在位 | 用户（Windows） |
| V9 | headless 冒烟（R4，根治判据二） | `start /wait "" dist\xstars\xstars.exe --help` + `echo %ERRORLEVEL%`（或无参数调用） | 优雅退出（exit code 1，Usage 语义）；**无** "Failed to execute script 'cli'..." 对话框 | 通过 | 用户（Windows） |
| V10 | 安装器产物验证（R5） | 核对 `installer\excel\output\XSTARS_Setup_v1.2.1.exe` 存在；`certutil -hashfile ... SHA256`；构建日志无 `WARNING: Expected ... not found` | 产物存在、文件名版本正确、哈希/大小已记录、无 WARNING | 通过 | 用户（Windows） |
| V11 | Excel 实机验证（R7） | 安装后真实 Excel 逐一点击 Quick Run / Run / Export High-Res | 无崩溃、分析与导出正常 | 全按钮通过 | 用户（Windows，本环境无法执行——需真实 Excel 宿主与人工操作，结果归档 PR 评论） |
| V12 | 包体积评估 | 对比 v1.2.0 与 v1.2.1 的 `XSTARS_Setup_v*.exe` 大小 | 增量约 15-30MB（问题简报估计值，**非硬阈值**） | 记录实测值即可（超过估计区间仅记录不阻断） | 用户（Windows） |

注：V3、V8–V12 不能在本环境执行，原因=需 Windows 开发机（本环境为 macOS）与真实 Excel 宿主，责任人=用户，按 M2 Runbook 执行并回传记录；V3 可与 V8 同批执行。

## 9. 文件级修改范围 + 风险 / 回滚 / 待决事项 + Git 策略

### 9.1 文件级修改范围

| 文件 | 操作 | 说明 |
| --- | --- | --- |
| `xstars.spec` | **修改** | import `collect_all`；`Analysis` 的 `binaries`/`datas`/`hiddenimports` 三处注入 `*pandas_*` 展开（T1.1）；其余部分不动 |
| `xstars/cli.py` | **修改** | 模块级新增 `import pandas  # noqa: F401` 预热（T1.2，仅 1 行 + 注释）；逻辑零改动 |
| `xstars/__init__.py` | **修改** | `:3` `__version__` → `1.2.1`（T1.3） |
| `pyproject.toml` | **修改** | `:7` `version` → `1.2.1`（T1.3）；依赖声明不动 |
| `installer/excel/XSTARS.iss` | **修改** | `:13` `MyAppVersion` → `1.2.1`，`:2` 头注释同步（T1.3）；Setup/Files/Registry/Code 段零改动 |
| `plans/20261007-windows-installer-pandas-fix.md` | **新建** | 本 Plan（规划产物） |
| `installer/excel/build_installer.py` | **明确不修改** | 既有三步流程满足 R4/R5（清理 :67-77、`--clean` :79-82、ISCC :251、产物名校验 :253-260） |
| `installer/wps/**`（含 `xstars-wps.spec`、`XSTARS_WPS.iss`、`wps_helper.py`） | **明确不修改** | WPS 链路隔离（Non-goal 2、R6） |
| `installer/mac/**`、`docs/macos-*`、`macos-pkg-acceptance-runbook.md` | **明确不修改** | macOS 链路隔离（Non-goal 1、R6） |
| `xstars/main.py`、`xstars/stats_engine.py`、`xstars/plot_engine.py`、`xstars/presets/**`、`xstars/application/**`、`xstars/wps_service.py` | **明确不修改** | 核心统计业务与 WPS 服务不动（Non-goal 3） |
| `ribbon/**`、`XSTARS_Templates.xlsx`、VBA | **明确不修改** | 与打包修复无关（Non-goal 8） |
| `tests/**` | **明确不修改** | 复用既有 `test_package_version_matches_pyproject` 做版本校验；冻结产物行为由 §8 V8–V11 命令行/人工验证承担（Non-goal 6） |
| `docs/windows-installer-build.md` | **默认不新建**（待决 D2） | Runbook 先随本 Plan §8/M2 交付；如需持久化为 docs 文档待用户裁定 |
| `requirements.txt` | **明确不修改** | 不动依赖（Non-goal 4） |

### 9.2 风险

| ID | 风险 | 等级 | 触发条件 | 影响 | 缓解 |
| --- | --- | --- | --- | --- | --- |
| RK-1 | 安装包体积增长 | 低 | `collect_all("pandas")` 全量收集（含 pandas 全部数据文件与 .pyd） | 安装包约 +15-30MB（问题简报估计值） | 对 Excel 完整客户端完全可接受（问题简报认定）；V12 记录实测值，不设硬阈值 |
| RK-2 | 构建环境缓存污染 | 中 | 旧 `build/`/`dist/`、PyInstaller 缓存或旧安装器混入，烟测对象非新包 | 验收结论不可信，缺陷漏网 | T2.1 强制清理 + `--clean`（build_installer.py:79-82）+ 删除 `installer/excel/output/` 旧产物；验证前核对产物时间戳与哈希 |
| RK-3 | 版本三源漂移（XSTARS.iss 无自动校验） | 中 | 只改 __init__/pyproject 忘 XSTARS.iss（任务书原文只列两处） | 产物名仍 v1.2.0，`build_installer.py:260` 仅 WARNING 不 fail | T1.3 三处同步；V5 + V10 双重兜底；D3 记录后续自动化方向（ISCC `/D` 注入） |
| RK-4 | Windows 构建环境漂移（pandas/PyInstaller 版本差异） | 中 | 开发机依赖版本与 v1.2 发布环境不一致 | 修复在特定组合下失效或不可复现 | T2.1 记录 `pip show pandas pyinstaller` 入 PR；烟测判据（V8/V9）环境无关 |
| RK-5 | `collect_all` 未根治（根因另有其他，如 hooks 干扰或 PyInstaller 版本缺陷） | 中 | V9 冒烟仍复现崩溃对话框 | 修复无效 | V8（二进制清单）+ V9（冒烟）为根治判据，失败即停在 M2、不打包发布；升级为独立调查并评估回滚（§9.3） |
| RK-6 | xstars.spec 同时被 macOS PyInstaller 注释路径引用 | 低 | 有人用该 spec 出 macOS 冻结包 | 该路径产物体积增大，无功能影响 | macOS 发布链路不走 PyInstaller（installer/mac/build_pkg.py；plans/20260906-macos-installer.md R17） |
| RK-7 | WPS 同类隐患（记录性风险） | 低 | `installer/wps/xstars-wps.spec` 同样未 `collect_all("pandas")`，若未来 WPS 链路触碰相同 pandas C-API 路径 | WPS 出现同类报错 | 问题简报认定 WPS 不受影响（spec 排除 xlwings，:51，崩溃前置条件不存在）；本次不改动（Non-goal 2），出现时复用 R1 方案另开任务 |
| RK-8 | Excel 实机验证依赖人工 | 低 | 本环境无 Windows/Excel | 最终验收闭环依赖用户回传 | V11 明确责任人=用户、记录归档 PR 评论；headless 判据（V8/V9）先行拦截大部分风险 |

### 9.3 回滚

- **合并前**：`git switch main && git branch -D fix/windows-installer-pandas-capi`；Draft PR 直接关闭。main 全程零改动，无需其他恢复动作。
- **合并后**：`git revert` 合并提交并重新出包（版本号按 D1 约定递增，如 1.2.2，避免同号重发混淆）；或保留修复、仅回退有问题的注入项（逐项 revert T1.1 的三处注入之一可作二分定位，配合 RK-5 调查）。
- **用户侧回退**：重新安装旧版安装器或卸载重装即可；本变更**无数据/配置/格式迁移**——仅打包配置 + 版本号 + 1 行防御性导入；`HKCU\Software\XSTARS\InstallPath` 注册表键（XSTARS.iss [Registry]）语义不变，卸载流程（`unregister_addin.vbs` / [UninstallRun]）零改动，用户工作簿与 `~/.xstars` 类用户数据不受影响。
- **平台兼容性**：macOS pkg 与 WPS 发行物零接触（Non-goals 1/2），无需任何回退动作。

### 9.4 待决事项

- **D1（实施前置）**：补丁版本号最终值——问题简报措辞为"**建议**升级为 1.2.1"。本 Plan 以 1.2.1 为默认值编排（R3、T1.3），**最终版本号需用户批准确认**；若裁定其他值（如 1.2.0-rebuild），T1.3 三处按裁定值同步。
- **D2**：Windows 构建/烟测 Runbook 是否需要持久化为 `docs/` 文档（如 `docs/windows-installer-build.md`）。本 Plan §8/M2 已交付完整步骤，默认不新建文档文件（Non-goal 与 §9.1 均按此编排）；如需持久化请另行裁定。
- **D3**（非决策，后续改进候选）：`installer/excel/XSTARS.iss` 版本自动化——ISCC 支持 `/D MyAppVersion=...` 由 `build_installer.py` 注入，可消除第三版本源人工同步面。本次不做（避免扩大改动面），列为残余改进项。
- **D4**（残余记录，非本 Plan 范围）：WPS spec 同类隐患（RK-7）——若 WPS 后续出现相同报错，复用 R1 方案另开需求。

### 9.5 Git 策略

- **分支名**：`fix/windows-installer-pandas-capi`（任务书显式指定的 PR 目标分支，覆盖默认 `feat/<slug>` 命名约定），自 `main` @ `b5705915` 创建。
- **PR 形态**：单一 **Draft PR**，base = `main`；M2 验收记录（环境版本、冒烟退出码、产物哈希、Excel 验证截图）以 PR 评论回传归档后转 Ready for review。
- **Draft PR 标题**：`fix(windows): collect full pandas payload in xstars.spec to fix frozen _pandas_datetime_CAPI crash`
- **PR 描述草稿**（可直接使用）：

  > ## Background
  >
  > XSTARS v1.2 on Windows crashes as soon as any ribbon button is clicked:
  >
  > ```
  > Failed to execute script 'cli' due to unhandled exception:
  > partially initialized module 'pandas' has no attribute '_pandas_datetime_CAPI'
  > (most likely due to a circular import)
  > ```
  >
  > The crash happens while `import_module("xlwings")` (xstars/cli.py:54) pulls in pandas
  > (xlwings/conversion -> pandas -> pandas/_libs/missing.pyx), i.e. **before** any workbook
  > is read (cli.py:56), so it is independent of the user's Office/system environment.
  >
  > Root cause: `xstars.spec` only hard-codes three `pandas._libs.tslibs` hiddenimports
  > (:78-81) with `binaries=[]` and no pandas data files, so PyInstaller on Windows does not
  > bundle pandas 2.x C-extension dynamic libraries (`pandas_datetime`/`pandas_parser` etc.)
  > and package data; pandas C-API initialization then fails inside the frozen app.
  >
  > macOS (python-build-standalone runtime, non-frozen) and WPS (separate spec,
  > `excludes=["xlwings"]`) are unaffected and untouched.
  >
  > ## Changes
  >
  > - `xstars.spec`: `collect_all("pandas")` -> inject full `datas`/`binaries`/`hiddenimports`
  > - `xstars/cli.py`: defensive `import pandas` prewarm before any xlwings import
  > - Version bump to 1.2.1 in `xstars/__init__.py`, `pyproject.toml`, `installer/excel/XSTARS.iss`
  >   (three version sources kept in sync; `test_package_version_matches_pyproject` guards the first two)
  >
  > ## Validation (Windows dev machine, recorded in comments)
  >
  > - [ ] `collect_all("pandas")` returns non-empty datas/binaries/hiddenimports
  > - [ ] `dist/xstars` contains `pandas_datetime*.pyd` / `pandas_parser*.pyd` / tslibs extensions
  > - [ ] headless smoke: `xstars.exe --help` exits gracefully (no "Failed to execute script" dialog)
  > - [ ] `XSTARS_Setup_v1.2.1.exe` built via Inno Setup; SHA256 recorded
  > - [ ] real Excel: Quick Run / Run / Export High-Res all work, no crash
  > - [ ] full pytest suite green (no new failures)

- **PR 拆分决策**：**单 PR，不拆分**。依据：改动面为 5 个文件（2 源码/配置 + 3 版本字面量）、同一故障主题、互相耦合（版本不同步会破坏 T2.3 产物校验）；拆分会导致中间态仍不可发布。
- **合并顺序**：M1（T1.1→T1.2→T1.3，可单提交或多提交串行进同一 PR）→ M2 Windows 验收记录回传（T2.1→T2.2→T2.3→T2.4）→ §8 V1–V12 全过后单次合并 `main` → 合并后由用户分发重打的 `XSTARS_Setup_v1.2.1.exe`（发布动作在本 Plan 外，随发布流程执行）。

---

## 完成前自查

- ✅ 9 段结构齐全且顺序固定（Goal / Requirements / Non-goals / Research summary / Gap analysis / Milestone 表格 / 分 milestone To-do / Validation contract / 文件级修改范围+风险/回滚/待决事项/Git 策略）。
- ✅ Milestone 数量 2 ≤ 10（目标 ≤7）。
- ✅ 每个功能缺口至少映射一个 To-do 任务 ID：**无孤立缺口**（G1→T1.1、G2→T1.2、G3→T1.3、G4→T2.2、G5→T2.1/T2.3、G6→T2.4；T1.4 为 M1 验证汇总，支撑 G1–G3，非范围外任务）。
- ✅ 每个任务均含文件、修改、验收、依赖四要素（T2.x 为执行型任务，"文件"字段注明产物路径与执行性质）。
- ✅ 所有歧义进入待决事项（D1 版本号最终值、D2 Runbook 持久化、D3 iss 版本自动化、D4 WPS 残余隐患），未擅自拍板产品决策；版本号以"建议 1.2.1 + D1 裁定"表述，未替用户定案。
- ✅ 输出仅写入合规 plans 路径 `plans/20261007-windows-installer-pandas-fix.md`，未创建或修改任何项目代码/测试/配置文件。
