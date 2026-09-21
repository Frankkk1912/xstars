# Explore: macOS 独立安装器（xstars）

- 日期：2026-09-06
- 主题：为 `xstars` 新增 macOS 独立安装器；验证 `xstars-dev` 中旧版资产的可用性与两仓库分叉影响面
- 性质：纯调研（零代码改动）。唯一写入即本报告
- 输入：`/feature-workflow` 路由 → `/feature-explore`
- 证据来源：`codebase-cartographer` 双 lane 调研（run `6e749efb`、`dbb2c5bc`）+ 父 Agent 锚点复核
- 锚点标注：〔P〕= 父 Agent 本次实读/实跑命令验证；〔C〕= cartographer 报告、父 Agent 未逐行复核（行号可能有 ±5 漂移，lane A 已观察到系统性漂移）

---

## 1. Implementation map

### 1.1 两仓库关系（已验证）

| 仓库 | HEAD | macOS 安装器现状 |
| --- | --- | --- |
| `/Users/frank/Documents/GitHub/xstars-dev` | `80c3465`（2026-05-07）〔P〕 | **有一整套 macOS 打包链路**（PyInstaller spec + .pkg 脚本 + 签名/公证脚本 + 已构建产物）〔P〕 |
| `/Users/frank/Documents/GitHub/xstars`（当前） | `ab30702`（main，PR #2 已合并）〔P〕 | **无任何 macOS 安装器**；`installer/` 下仅 `wps/`（Windows）〔P〕；文档明确 `.app`/签名/公证不是当前交付物〔P〕 |

结论：用户记忆成立——旧版确实存在且已实装到"能产出 .app 与 .pkg"的程度；但**旧版从未完成签名与公证**，且其运行时桥接方式与当前 main 架构已不兼容（见 §2、§5）。

### 1.2 旧版安装链路（三层，`xstars-dev`）

```text
构建期  PyInstaller(xstars.spec, IS_MAC 分支)
          └─> dist/XSTARS.app  (onedir + BUNDLE, LSUIElement=True 〔P spec:171〕)
        installer/build_installer.py  (--target=auto|win|mac; --skip-pyinstaller/--skip-xlam/--skip-package) 〔C〕
          └─> bash installer/mac/build_pkg.sh   〔C build_installer.py:296〕
                ├─ tar -czf XSTARS.app.tar.gz        # 规避 Sequoia com.apple.provenance → AppleDouble 丢目录 〔C mac_porting_notes.md:144-187〕
                ├─ osacompile xstars_launch.applescript -> .scpt
                ├─ pkgbuild  -> XSTARS-component.pkg
                ├─ productbuild --distribution distribution.xml(sed __VERSION__) -> installer/output/XSTARS-{ver}.pkg
                └─ if XSTARS_SIGN=1 -> sign_and_notarize.sh        〔P build_pkg.sh:97-100〕

安装期  Installer(root) -> installer/mac/postinstall.sh
          ├─ rm -rf /Applications/XSTARS.app; tar -xzf -> /Applications/; chown root:wheel; chmod 755 二进制  〔P postinstall.sh:27-31〕
          ├─ cp XSTARS.xlam -> /Applications/XSTARS/XSTARS.xlam                                            〔P :34-37〕
          ├─ 探测 console 用户 -> cp XSTARS.xlam -> ~/Library/Group Containers/UBF8T346G9.Office/User Content.localized/Startup.localized/Excel 〔P :48-50〕
          └─ sudo -u <user> mkdir -p ~/Library/Application Scripts/com.microsoft.Excel && cp xstars_launch.scpt  〔P :65-69〕

运行期  Excel 加载 Startup/XSTARS.xlam -> Ribbon -> VBA RibbonCallbacks.RunCmd
          ├─ ExePath() = /Applications/XSTARS.app/Contents/MacOS/xstars
          ├─ shellCmd = ('.../xstars' run_quick '<wb>' </dev/null >/dev/null 2>&1 &)
          └─ AppleScriptTask "xstars_launch.scpt" "xstarsLaunch" shellCmd   〔C ribbon_callbacks_installed.bas:21-27, 34-45〕
                └─> 子进程 xstars.cli:main() -> xw.Book(path) -> Tk/ttkbootstrap（强制 -topmost）
```

### 1.3 当前仓库 macOS 运行形态（已验证，与旧链路**不同**）

- 入口：Excel 侧走 **xlwings `RunPython`**（`ribbon/ribbon_callbacks.bas`，`RunPython` 命中 36 次〔P〕）；`ribbon_callbacks_installed.bas` 已是 **Windows-only**（`Mac`=0、`RunPython`=0、`Shell`=4、读 `HKCU\Software\XSTARS\InstallPath`、启动 `xstars.exe`〔P 头 20 行〕）。
- 文档契约：`docs/macos-developer-setup.md:3` "This is not a standalone macOS application."；`:5` "does not provide a `.app`, DMG, signing, or notarization workflow."〔P〕
- 运行时假定：本地 venv + `pip install -e ".[dev]"`，xlwings 解释器手工指向 `.venv/bin/python`〔P docs:41-64 区间〕。
- 冻结态支持：`xstars/wps_service.py:303` **已有** `getattr(sys, "frozen", False)` 分支〔P〕（推翻 lane B "全库无 sys.frozen" 的结论），但仅限 WPS 服务自重启路径；核心 Excel 链路无冻结资源解析（全库无 `sys._MEIPASS`/`importlib.resources`〔P grep 无命中，仅 .pyc 噪声〕）。
- 配置与工件根：`xstars/config.py:10` `DEFAULT_SETTINGS_PATH = ~/.xstars/settings.json`〔P〕；`xstars/artifacts.py:47` `DEFAULT_ARTIFACT_ROOT = ~/.xstars/artifacts`〔P〕。
- 新增运行时组件（旧版无）：`xstars/wps_service.py`（回环 HTTP，`127.0.0.1:3892`，端口集 `frozenset({3889, 3890})`〔P :759〕）、`xstars/application/`、`xstars/artifacts.py`。
- 扩展点：`sys.platform == "darwin"` 平台分流已在导出链路成型〔C main.py:1330-1349〕；Tk 错误框带回退〔C main.py:310-352〕；macOS 选区输入框〔C main.py:1759-1823〕。
- CI 面：`.github/workflows/macos-support.yml` 三 job 分别在 `macos-latest` / `windows-latest` / `ubuntu-latest`〔P :16,41,66〕；触发 `pull_request`（全量）+ `push: feat/macos-support`〔P :3-7〕。**无签名/公证 job，无 artifacts 发布，无证书 secrets**〔P 工作流仅此一个文件〕。

### 1.4 两代架构对比（已验证）

**旧版 `xstars-dev@80c3465`：扁平两层 + 单一“冻结 exe 子进程”模型（`xstars/` 共 6,395 行）**

- 子包仅 `presets/`、`tools/`，**无 `application/`**；`main.py` 1,543 行在模块顶部直接 `import xlwings as xw`（`main.py:7`）→ 宣主 API 与业务编排同文件，无宣主无关层。
- `cli.py` 51 行，**单一模式**：`argv[1]=command`、`argv[2]=workbook_path` → `xw.Book(path).set_mock_caller()` → `main.<command>()`；docstring 首行即 “CLI entry point for frozen (PyInstaller) distribution. VBA calls via Shell”〔P〕。
- 图导出持久化：`_export_cache.py` 70 行，**pickle** matplotlib Figure 到 `%TEMP%/xstars/fig_cache`（md5(workbook::picture) 键、7 天 GC）；docstring 明写设计前提 “The frozen exe is launched fresh by VBA on every ribbon click, so an in-memory cache wouldn't survive”〔P〕。
- **Python 侧零平台分支**：全仓 `xstars/*.py`、`presets/`、`tools/`、`ui_dialog.py` grep `darwin|sys.platform` 均无命中〔P〕→ macOS 适配全部落在 Python 之外（`.bas` 的 `#If Mac`、自研 `xstars_launch.scpt`、Tk topmost、打包脚本）。

**当前 `main@ab30702`：三层 + 三宣主三进程模型（`xstars/` 共 11,289 行）**

| 层 | 模块（行数） | 职责 |
| --- | --- | --- |
| 共享内核（宣主/平台无关） | `plot_engine` 497、`stats_engine` 326、`styles` 391、`data_handler` 172、`config` 268、`presets/`、`tools/`、**`artifacts.py` 914** | 统计/绘图/配置/图重建工件 |
| 宣主无关应用层 `application/` | `contracts.py` 510、`analysis.py` 934、`export.py` 372、`worker.py` 761 | JSON-safe DTO + `Command` 白名单枚举（明写 “exclude callables, pickle payloads, shell fragments”）；worker docstring：“Any Tk dialog is created on the worker's main thread; the HTTP server never imports or creates GUI objects” |
| 宣主适配层 | `main.py` 1,965（Excel/xlwings）、`wps_service.py` 808（WPS HTTP broker） | 宣主 API 与进程模型 |

- **分层方向已验证干净**：`application/*` 不 import `main`/`wps_service`/`xlwings`（反向依赖 grep 为空〔P〕）；宣主层通过**动态 import** 下沉调用（`main.py:26` `xw = import_module("xlwings")`、`main.py:27` `_application_analysis = import_module(".application.analysis", __package__)`〔P〕），`wps_service.py:22` 静态 import `contracts`。
- **三宣主三进程模型**：Excel/Windows = 冻结 exe 子进程（VBA `Shell` → `xstars.exe`，`cli.py` 默认分支）；Excel/macOS = **`RunPython` 同进程**（xlwings 官方桥，`ribbon_callbacks.bas` → `xstars.main.run*()`）；WPS/Windows = HTTP broker + 一次性 worker 子进程（`cli.py` `serve`/`worker` 模式）。三者共用 `artifacts.py`（被 `main.py`、`wps_service.py`、`application/export.py` 引用〔P〕）。
- **平台分支收敛度**：`darwin` 仅出现于 `main.py`（8 处〔P〕），内核与 `application/` 层零平台分支——这是旧版不具备的性质。

**对安装器的直接含义**：旧安装器不能直搜的根因不是脚本质量，而是它服务的进程模型（冻结 exe + VBA Shell 子进程 + pickle 跨进程传图）在当前 macOS 契约（`RunPython` 同进程 + `ribbon/*.bas` 零修改）下不成立。选定方案（自带非冻结 Python）恰好是当前架构的**原生形态**：业务代码一行不改，只把“venv 从哪来”换成“包内自带”。

### 1.5 两代架构优劣判定（Plan 阶段的方案依据）

**结论：当前这代明显更优，且四个维度均有可验证硬数据，不是风格偏好。**

| 维度 | 旧版 `xstars-dev@80c3465` | 当前 `main@ab30702` | 判定依据 |
| --- | --- | --- | --- |
| 宣主解耦 | `main.py:7` 顶部直接 `import xlwings`，业务与 Excel API 同文件 | `application/` 2,577 行宣主无关层；反向依赖 grep 为空（不 import `main`/`wps_service`/`xlwings`）〔P〕 | WPS 支持能作为独立 PR 加入靠的就是这层；旧架构下支持 WPS ≈ 重写 `main.py` |
| 平台差异可测性 | Python 侧 `darwin`/`sys.platform` **0 处**；tests 11 文件 1,900 行，`darwin` **0 处**〔P〕 | `darwin` 收敛在 `main.py` 8 处；tests **40 文件 7,458 行**、`darwin` **61 处**，含专门 `tests/test_macos_support.py`（12+ 个 `test_darwin_*`）〔P〕 | **差距最大项**。旧版把平台差异全推到 Python 之外（`.bas` `#If Mac`、自研 `.scpt`、打包脚本）→ 一行都测不到，只能真机踩坑（`mac_porting_notes.md` 328 行就是代价账单） |
| 持久化安全 | `_export_cache.py`：**pickle** Figure 到 `%TEMP%/xstars/fig_cache`，md5 文件名 + 7 天 GC，**零完整性校验**〔P〕 | `artifacts.py` 914 行：`SCHEMA_VERSION`(:46)、`manifest.json`(:48)、`_SAFE_KEY = ^[0-9a-f]{64}$`(:49)、malformed/integrity 异常类(:75)、`_atomic_write_json`(:223)、payload 权威 + unlink-first（:183-184 明写“manifest 竞争失败不能污染 payload”）〔P〕 | 安全级别差异：pickle 反序列化 = 任意代码执行面，而 `%TEMP%` 全局可写。`contracts.py` docstring 直写 “exclude callables, **pickle payloads**, shell fragments” |
| 复杂度是否有归处 | 6,395 行 | 11,289 行（+77%），但 `main.py` 仅 1,543→1,965（+27%）；新增 ~4,900 行中 `application/` 2,577 + `artifacts` 914 + `wps_service` 808 = 4,299 行均为新能力的独立归处〔P〕 | 不是“更臃肿”而是新能力未被堆进上帝文件；测试/代码比 0.30 → **0.66** |

#### 旧版仍不输的三处（诚实记账）

1. **交付维度它今天真的领先**：盘上已有可装的 `XSTARS-1.1.1.pkg`（66.6 MB）与 `XSTARS.app`（154 MB），新版这块为零——这是它唯一实质优势，也正是本轮要移植的东西。
2. **一处真实退化 + 一处设计取舍**：旧 `cli.py` 有 `xw.books[...]` 未保存工作簿回退，新 `cli.py:56` 直接 `xw.Book(path)` 丢了它（退化）；但新版导出路径改为显式友好报错并有测试锁定（`test_darwin_unsaved_workbook_export_reports_save_and_regenerate`），那是 fail-closed 哲学的取舍，不算退化。两者必须分开看。
3. **认知负担确实变高**：`main.py` 1,965 行 + 大量 `import_module` 字符串导入，IDE 跳转/静态分析/打包工具都更难伺候。动机部分可见（`_ttkb = import_module("ttkbootstrap")` 配 `try/except ImportError` 做可选依赖降级，合理），但 `xw = import_module("xlwings")`（`main.py:26`）并无降级需求，属可议写法 → **技术债**。分层本身也产生过债：DC1 去重（`application/analysis.py` 内被复制一份 `presets/qpcr.py` 实现）已由 PR #5 修掉。

**对本轮的指导意义**：本轮不是“选哪代架构”，而是**把旧版唯一领先的那部分（能出包的 `.pkg` 链路）移植到新架构上，同时不复活它的进程模型**（冻结 exe + VBA Shell 子进程 + pickle）。选定的 B1 不需为打包牺牲新架构任何性质：动态 import 陷阱自动消失（真解释器 + 真 site-packages）、`application/`与`artifacts.py`原样受益、`darwin` 分支仍只在 `main.py` 因而仍可测、`ribbon/*.bas` 零改动保住 CI 门禁。反之，若为复活旧打包而回到旧进程模型，等于用“零平台测试覆盖 + pickle 缓存 + 宣主耦合”换回一个从未签名分发过的 `.pkg`，交换不划算。

---

## 2. Key files and symbols

### 2.1 旧仓库可复用资产（`xstars-dev`，行数为〔P wc -l 实测〕）

| 文件 | 实测行数 | 关键锚点 | 与主题关系 |
| --- | --- | --- | --- |
| `installer/build_installer.py` | 371〔P〕 | 路径常量 40-45；`--target` 311-318；`ensure_xlam_for_mac()` 152-171；`bash build_pkg.sh` 296〔C〕 | 跨平台打包调度入口；含大量旧布局硬编码 |
| `installer/mac/build_pkg.sh` | 104〔P〕 | tar 规避 AppleDouble 40-44；osacompile 56；pkgbuild 63-69；productbuild 82-85；签名触发 97-100〔P〕 | .pkg 主流程，核心可用 |
| `installer/mac/distribution.xml` | 29〔P〕 | `hostArchitectures="arm64,x86_64"` **:11**〔P〕；`<os-version min="11.0">` **:14**〔P〕；`__VERSION__` 宏 | 修掉 Rosetta 降级；真实被 productbuild 使用 |
| `installer/mac/postinstall.sh` | 80〔P〕 | 27-31 / 34-37 / 48-50 / 65-69〔P〕 | root 期部署编排：.app + xlam + scpt |
| `installer/mac/sign_and_notarize.sh` | 81〔P〕 | env 读取 25-27；空值即 `exit 0` 29-41〔P〕；codesign `--deep --force --options runtime` **:53**（**无 `--entitlements`**〔P〕）；productsign :66；notarytool :70；stapler :76-77〔P〕 | 签名/公证：调用真实存在，但凭据全为空 + 递归缺陷 |
| `installer/mac/xstars_launch.applescript` | 25〔P〕 | `on xstarsLaunch(shellCmd)` 22-25〔C〕 | 沙盒逃逸桥（VBA `AppleScriptTask` 目标） |
| `xstars.spec` | 178〔P〕 | `Analysis(["xstars/cli.py"])`；`xlwings_backend` 38-47；**`"xstars._export_cache"` :102**〔P〕；`bundle_identifier="com.frank-sysu.xstars"` **:163**〔P〕；`LSUIElement: True` **:171**〔P〕；`icon=None`；无 `entitlements_file` | 双平台 spec；mac 分支结构完整但清单已过时 |
| `dist/XSTARS.app` | 154 MB | `Contents/MacOS/xstars`；`CFBundleIdentifier=com.frank-sysu.xstars`；`CFBundleShortVersionString=1.1.1`；`LSUIElement=true`；Resources 内**无** xlam/xlsm/templates | 旧产物；实测 `codesign -dv` → **`Signature=adhoc`、`TeamIdentifier=not set`、Mach-O **thin (arm64)**〔P〕 |
| `installer/output/XSTARS-1.1.1.pkg` | 66.6 MB | `spctl -a -t install` → **`rejected, source=no usable signature`**〔P〕 | 旧 .pkg **从未签名/从未公证** |
| `mac_porting_notes.md` | 328〔P〕 | 9 类已踩坑（xlam 无法自动构建、AppleDouble、Rosetta、`MacScript` 封锁、`open -n -a` 静默失效、未保存工作簿、`LSUIElement` 焦点）〔C〕 | 最高价值经验资产 |
| `ribbon/ribbon_callbacks_installed.bas`（旧） | 215〔P〕 | `#If Mac` 21-27, 34-45〔C〕 | 旧 Mac 唤起模型（当前仓库已无此分支） |

### 2.2 当前仓库约束点（`xstars`）

| 锚点 | 事实 |
| --- | --- |
| `pyproject.toml:7` = `version = "1.1.1"` 〔P〕 vs `xstars/__init__.py:3` = `__version__ = "1.0.0"` 〔P〕 | **版本双源且不一致**；旧 `build_installer.py:51` 从 `xstars/__init__.py` 取版本 → 直接移植会产出 1.0.0 的包 |
| `pyproject.toml` 无 `[project.scripts]`/`[project.gui-scripts]`；`[tool.setuptools.packages.find] include = ["xstars*"]`，无 package-data 〔C，父 Agent 未逐行复核〕 | 无脚本入口可挂靠；模板/资源不在包数据内 |
| `docs/macos-developer-setup.md:3,5`〔P〕 | 现契约：非独立应用、无 .app/签名/公证 |
| `docs/macos-developer-setup.md:29-37`〔P 区间存在；具体表述未逐字复核〕 | 实机结论：Mac Excel 不渲染 `customUI14`（2009 命名空间），必须 2006-format `.xlam` + Startup 目录 |
| `docs/macos-developer-setup.md:79-84`〔P 区间；`:79` 附近〕 | "reuses `ribbon/ribbon_callbacks.bas` unchanged … Do not create a separate Mac callback module" |
| `plans/20260829-macos-support.md:24`〔P〕 | R1：独立 `.app` 分发**留待后续 PR** |
| `plans/20260829-macos-support.md:46,57-59`〔P〕 | R12 不新建/改 AppleScriptTask VBA；Non-goal 1 不构建 .app/PyInstaller/universal2/DMG/Homebrew；Non-goal 2 不做 Developer ID/Hardened Runtime/entitlements/notarization；Non-goal 3 属"后续独立分发 PR" |
| `docs/cross-platform-office-technology-strategy.md:85`〔P〕 | 实施原则 6："Windows 与 macOS 使用独立安装包和签名流程"（即 memory 所称"决策点 6"的真实位置） |
| `plans/explore-20260829-macos-compat.md:251`〔P〕 | 遗留开放问题：分发渠道 .app / Homebrew tap / pip |
| `xstars/cli.py:56` `book = xw.Book(workbook_path)`〔P〕 | **无** 旧版 `xw.books[...]` 未保存工作簿回退（旧 `cli.py:34-40` 有） |
| `.github/workflows/macos-support.yml`〔P〕 | 已有 `macos-latest` job；无签名/发布 job |
| 全仓库 `find . -name "*.xlam"` = 空〔P〕 | 当前仓库**不持有** .xlam 资产 |
| 全仓库 `find . -name "*.entitlements"` = 空〔P〕 | 无 entitlements 资产 |

### 2.3 当前仓库对标模板（`installer/wps/`，Windows 侧既有约定）

- 目录形态：`installer/<product>/{<product>.spec, XSTARS_<PRODUCT>.iss, build.ps1, <helper>.py}`〔P〕
- 版本注入：`XSTARS_WPS.iss:4-5` `#ifndef AppVersion / #define AppVersion "1.0.4"` → 构建期由 ISPP define 覆盖；`build.ps1:85-94` 从 `wps-addon/package.json` 读版本〔P〕。**约定 = 版本在构建期单一来源注入，.iss 内仅兜底默认值**
- 安装内容：`Source: "dist\xstars-wps\*"`、`wps-addon/deploy/*`、`config.template.js`、`docs/wps-installation.md`〔P iss:33-38〕；单用户安装、免提权、默认 `%LOCALAPPDATA%\XSTARS-WPS`〔C docs/wps-installation.md:34〕；安装器负责后台自启动服务并继承 `~/.xstars/wps_service.json` 令牌与端口〔C docs/wps-installation.md:43,79〕

---

## 3. Existing patterns and conventions

1. **平台分流靠 `sys.platform == "darwin"`**，不散布 try/except 猜测——`main.py` 导出分流〔C:1330-1353〕、`wps_service.py:303` frozen 判定〔P〕。
2. **fail-closed 优于回退**：macOS 导出无 artifact 时直接报错，不回退截图〔P docs:129-146 区间；C explore-20260829〕。
3. **UI 层级问题在代码层修**：Tk 一律 `root.attributes("-topmost", True)`（`main.py:322`〔C〕；旧仓 `mac_porting_notes.md:308-315` 同结论）。
4. **安装器 = 目录 + 构建脚本 + 打包规格三件套**，按产品线分目录（`installer/wps/`），不共享跨产品安装包〔P；docs/wps-support-implementation-plan.md:30 "WPS 使用独立安装包，不与 Excel 安装器合并"〕。
5. **版本由构建脚本注入，不在安装脚本里手写**〔P iss `#ifndef` define + build.ps1 读 package.json〕。
6. **文档即验收契约**：`docs/macos-manual-acceptance.md:3-5`〔C，区间存在〕明确"CI 不启动 Excel、不弹 Tk、不碰 Automation 授权"→ 真机验收由人执行并回填 Draft PR（`plans/20260829-macos-support.md:18`〔P〕）。
7. **`ribbon/*.bas` 零修改是 CI 硬门禁**（不是文档约定）：`.github/workflows/macos-support.yml:65` job “Existing VBA files are unchanged” → `:73` `git diff --exit-code origin/main...HEAD -- 'ribbon/*.bas'`〔P〕。任何 PR 改 .bas 即变红；另外 plans/20260829-macos-support.md:24,46 与 plans/20260906-merge-main-macos-support.md:44-46 把它写为交付约束。
8. **测试/lint**：pytest + ruff（`.ruff_cache/0.16.5` 存在于 `installer/wps/`、`poc/wps/`〔P〕）；本报告未展开 tests/ 结构 —— **证据缺口**。

---

## 4. Candidate implementation paths

> 只梳理候选，不作最终选择。

### 路径 A：旧资产直译移植（最小改动复活旧链路）

- **依托**：整搬 `xstars-dev/installer/mac/*` + `xstars.spec` → 新仓库 `installer/excel-mac/`；沿用 `postinstall.sh` 的三处部署 + `AppleScriptTask` 自研 `.scpt`。
- **旧 xlam 生产方式（已查清）〔P〕**：`xstars-dev/installer/build_installer.py:105-146` `build_xlam_win()` 在 **Windows** 上用 Excel COM 把 `ribbon/ribbon_callbacks_installed.bas` 注入新建工作簿并 `SaveAs(..., 55)` 存为 .xlam，注称“.xlam is then committed to the repo and reused on Mac builds (it's platform-agnostic)”→ 旧项目的约定是 **xlam 作为入库二进制制品 + Windows 侧生成**；而当前仓库的 `installed.bas` 已无 Mac 分支〔P〕，照旧注入得到的 xlam 在 Mac 上必挂（`WScript.Shell`/`RegRead`）。
- **必须处理**：spec 清单重写（删 `_export_cache` :102，补 `application/`、`artifacts`、`PIL`、`wps_service`）；xlam 必须改为 2006-format 且需人工构建；**必须在 `ribbon_callbacks_installed.bas` 重新引入 `#If Mac` 分支** → 直接冲撞零修改硬门禁〔P docs:79-84；plans:46〕。
- **代价/复杂度**：低-中（脚本本体可用）。**兼容性风险：高**（VBA 契约 + 现文档承诺 + 当前 main 已删除该分支）。
- **产物形态**：除非具备证书，仍是 adhoc/unsigned → 本机实测 `spctl` rejected〔P〕，终端用户会被 Gatekeeper 拦。
- **适用条件**：仅内部/自用分发，且愿为 Mac 单开一条 installed-mode VBA 交付线。

### 路径 B：只替换“运行时来源”，保留 `RunPython` 桥（半独立安装器）

- **依托**：现有 `ribbon/ribbon_callbacks.bas`（`RunPython` 命中 36 次〔P〕）**零改动** + xlwings 官方 macOS 脚本支持（`xlwings runpython install` 已自行完成沙盒逃逸，**不需要**我们自建 `.scpt`〔P docs:63-67,197〕）；`.pkg` 只负责投递自带 Python 运行时 + 指向解释器 + 部署 2006-format xlam 到 Startup。
- **子变体必须区分（本次补充的关键约束）**：
  - **B1 内置非冻结 Python**（python-build-standalone / 携带一个真实 `bin/python` + 已装依赖的 site-packages）→ 把 xlwings `Interpreter` 指到它（工作簿 `xlwings.conf` 或用户配置〔P docs:96〕），`RunPython` 照常可用 → **`ribbon/*.bas` 零改动，CI 门禁不触发**。
    - **附加优势（已验证）**：当前代码大量使用字符串式动态 import（`main.py:26,27,30,452,1097-1099,1688`：xlwings / ttkbootstrap / tkinter / matplotlib.pyplot / `.application.analysis`；`cli.py` 同理 import `xstars.wps_service`、`xstars.application.worker`〔P〕）。静态分析看不见这些依赖 → 对 PyInstaller 是 hiddenimports 陷阱；**而 B1 用真解释器 + 真 site-packages，动态 import 天然正常，这个陷阱直接消失**。
  - **B2 PyInstaller 冻结 .app** → `RunPython` 需要真实解释器，不能指到冻结二进制（需 `RunFrozenPython` 或子进程，而两者都被 `plans/20260829-macos-support.md:46,59` 排除）→ **必然要求改 .bas 或改工作簿调用方式，即退回路径 A**。
  - 推论：“不破 ribbon 零修改”与“PyInstaller 产物”互斥；事定选项。
- **主要影响面**：`installer/excel-mac/`（新增）、解释器配置写入、`docs/macos-developer-setup.md` 需新增"非开发者模式"章节；`sys.frozen` 资源解析需补齐（当前仅 `wps_service.py:303`〔P〕）。
- **代价/复杂度**：中。**不破任何既有硬门禁**，是三条路径中与当前 main 契约冲突最小者。
- **关键取舍**：`RunPython` 仍依赖工作簿/加载项内 `xlwings.conf` 或全局配置指向内置解释器 → 多版本共存与升级路径需设计；未签名仍是 Gatekeeper 阻断点。

### 路径 C：产品化独立分发（对齐战略原则 6〔P strategy:85〕）

- **范围**：`.app` + `.pkg` + Developer ID Application/Installer + Hardened Runtime + `*.entitlements` + notarytool 提交 + `stapler` + 卸载/`MicrosoftRegistrationDB` 清理 + universal2 + CI 签名 job。
- **硬前置（本机实测）**：`security find-identity -v -p codesigning` → **0 valid identities found**〔P〕；`notarytool --keychain-profile xstars` → **No Keychain password item found**〔P〕；仓库无 `.entitlements`〔P〕；CI 无 mac 发布 job〔P〕。
- **不可自动化的阻断点**：`.xlam` 需人工在 Mac Excel VBE 导入构建（`mac_porting_notes.md:53-118`〔C〕；当前仓库已不入库 .xlam〔P〕）→ CI 无法端到端出包，除非改为"预构建 xlam 入库"或改用不依赖 VBA 的注册形态。
- **代价/复杂度**：高（证书、secrets、双架构、真机验收、发布流程）。**唯一能达成"零 Python、双击可用"目标的路径。**
- **适用条件**：确认要对外发布；接受先建 xlam 制品治理与签名凭据体系。

---

## 5. Preliminary gap analysis（功能缺口初稿，非任务拆解）

| 目标能力 | 现状 | 证据 | 影响 |
| --- | --- | --- | --- |
| macOS PyInstaller spec 在当前仓库存在 | **无** | 当前仓库无 `xstars.spec`；旧 `xstars.spec:1-178`〔P〕 | 无法构建 .app；旧 spec 需整体重写清单 |
| hiddenimports 与新架构匹配 | **过时/致命** | 旧 spec:102 引已删除的 `xstars._export_cache`〔P〕；缺 `application/`、`artifacts`、`PIL` | 构建期或运行期 `ModuleNotFoundError` |
| 随包模板资产 `XSTARS_Templates.xlsx` | 未收集 | 旧 spec datas 仅 matplotlib/ttkbootstrap/statannotations〔C:50-60〕 | 装机后无示例模板，README 承诺落空 |
| `.xlam` 加载项制品 | **两仓库均不可自动产出（所有路径共同的硬前置）** | 当前仓库 `find *.xlam` 空〔P〕；旧仓由 `build_installer.py:105-146` 在 **Windows COM** 生成并入库〔P〕；Mac 无法脚本构建（`mac_porting_notes.md:53-118`〔C〕；`docs/macos-developer-setup.md:33` 亦要求“在 Windows 或 Office RibbonX Editor 侧打包好再拷到 Mac”〔P〕） | 安装器无 UI 入口可投递；CI 无法端到端闭环（需跨机制品治理） |
| customUI 命名空间正确性 | 旧链路错 | 旧 `build_installer.py:160-200` 注入 2009；`docs/macos-developer-setup.md:29-37`〔P〕实测必须 2006 | 旧包在 Mac Excel 功能区不渲染 |
| VBA → 独立二进制的唤起桥 | 当前 main 已剥离 | 新 `installed.bas` Mac=0〔P〕；旧 21-27/34-45〔C〕；`xstars_launch.scpt` 当前仓库不存在〔P〕 | 走 A 需破零修改门禁；走 B 需依赖 xlwings 桥 |
| 自带 Python 运行时的资源解析 | 基本无 | 全库无 `_MEIPASS`/`importlib.resources`〔P〕；仅 `wps_service.py:303` frozen 分支〔P〕 | 冻结后找不到模板/配置/工件根 |
| 未保存工作簿回退 | 新版丢失 | 新 `cli.py:56` 无 `xw.books[...]`〔P〕；旧 `cli.py:34-40` 有〔C〕 | 新建未存工作簿点击按钮 → 子进程静默崩溃 |
| 后台 .app 与 Tk 焦点 | 已知冲突未解 | 旧 spec:171 `LSUIElement=True`〔P〕；`mac_porting_notes.md:308-315`〔C〕 | .app 静默启动时 Tk 窗被 Excel 遮挡、键盘不可用 |
| 代码签名（Developer ID） | **完全不具备** | 本机 `security find-identity` = 0〔P〕；旧 `.app` adhoc〔P〕；`.pkg` spctl rejected〔P〕 | 分发即被 Gatekeeper 拦 |
| 公证链路与凭据 | 骨架但不可用 | `sign_and_notarize.sh:25-41` 空 env 即 `exit 0`〔P〕；无 notary keychain profile〔P〕 | 任何"看起来已签名"的产物都不存在；文档化步骤缺失 |
| 签名脚本逻辑正确性 | **缺陷** | `build_pkg.sh:97-100` 触发 sign；`sign_and_notarize.sh:60` 无条件回跑 `build_pkg.sh`〔P〕，`XSTARS_SIGN=1` 会随环境继承 → **无限递归**；`codesign --deep` 无 `--entitlements`〔P :53〕 | 不能原样复用，必须改单向流水线 |
| entitlements 资产 | 无 | 仓库无 `.entitlements`〔P〕 | Hardened Runtime 下 dylib/JIT/网络可能受限（含 `wps_service` 端口绑定） |
| 双架构 | 未做 | 旧产物 `Mach-O thin (arm64)`〔P〕；Non-goal 明确排除 universal2〔P plans:57〕 | Intel 用户不可用 |
| 卸载 / 升级生命周期 | 无 | 旧仅覆盖式 `rm -rf`〔P :27〕；`pkgutil --forget`/`MicrosoftRegistrationDB` 残留需手工（memory 实证 + docs:218-223） | 卸载后 Excel 每次启动报"找不到加载项" |
| 版本单一来源 | **冲突** | `pyproject.toml:7`=1.1.1 vs `xstars/__init__.py:3`=1.0.0〔P〕 | 包名/收据/文档版本可能互相矛盾 |
| CI 出包与发布 | 无 mac 发布 job | `macos-support.yml` 仅测试〔P〕 | 构建不可重复、需本地手工出包 |

---

## 6. Open questions（进入 `/feature-plan` 前需你裁定）

### 已裁定（2026-09-06 用户确认，Plan 直接采用）

- **Q2 签名/公证：本轮不做**。产物为未签名 `.pkg`（本机自装无感；跨机分发需右键打开/去 quarantine，待 Plan 阶段核实 Sequoia 具体行为并写入文档）。
- **Q7 架构：arm64-only**（不做 universal2；`distribution.xml:11` 现有 `arm64,x86_64` 声明需改为 `arm64`）。
- **Q3 包体形态：`.pkg`**（能跑 postinstall 自动配 xlwings，不用 `.dmg`）。
- **Q5 xlam 制品：接受入库二进制**（`XSTARS.xlam` + 预置 `.xlsm` 模板，Windows/RibbonX Editor 侧一次性生成后提交）。

### 范围与定位

1. 交付目标是 **C 面向终端用户的"零 Python、双击可用"**，还是 **B 只把 Python 运行时自带的增强开发者模式**？两者对签名是"必需"vs"可选"的分水岭。
2. 本轮是否包含 **Developer ID 签名 + 公证**？若包含：你是否已有/将注册付费 Apple Developer 账号？本机当前 **0 张有效签名身份**〔P〕，这是硬阻断。
3. 分发渠道：`.pkg`、`.dmg`+`.app`、Homebrew tap，还是继续 `pip install -e`？（`plans/explore-20260829-macos-compat.md:251` 遗留未决〔P〕）

### 契约冲突（必须显式破或立）

1. 是否允许**新增 Mac 专用 VBA 唤起分支**（破 `docs/macos-developer-setup.md:79-84` 与 CI 零差异门禁〔P〕）？不允许则只能走 xlwings `RunPython` 桥。
2. `.xlam` 从哪来？（a）手工构建后**入库为二进制制品**；（b）构建期由脚本生成（Mac 上不可行〔C〕）；（c）不再依赖 xlam，改 xlwings addin + 用户工作簿。当前仓库不持有任何 `.xlam`〔P〕。
3. `com.frank-sysu.xstars` bundle id 与旧 `pkgutil` 收据 `com.frank-sysu.xstars` 是否继续沿用？（旧机器上已存在该收据与卸载残留问题）

### 行为与验收

1. 最低支持 macOS 与架构：`distribution.xml:14` 现为 11.0、`:11` 为 `arm64,x86_64`，但旧产物实为 **arm64 thin**〔P〕。本轮是否要求 universal2？（Non-goal 57 现排除它〔P〕）
2. 未保存工作簿、Tk 焦点遮挡、`LSUIElement` 是否算本轮必修（旧笔记均记为已知坑〔C〕）？
3. 卸载是否需一等公民（含 `MicrosoftRegistrationDB` 残留清理与 Excel 弹窗消除）？
4. 真机验收由你在 Mac mini 执行并回填 Draft PR —— 是否沿用该门禁〔P plans:18〕？
5. 版本号单一来源裁定：以 `pyproject.toml` 为准并把 `xstars/__init__.py:3` 对齐到 1.1.1，还是反向？〔P 双源冲突〕

### 治理

 1. `xstars-dev` 旧仓库如何处置（归档只读 / 明确废弃 / 作为资产来源保留）？`mac_porting_notes.md` 是否迁入当前仓库 `docs/`？
 2. CI 是否需要新增 mac 出包 job？若证书无法进 CI，是否接受"CI 只跑测试、出包在本机"的显式记录？

### 证据缺口（调研未覆盖，Plan 阶段需补）

- `tests/` 对打包/配置路径的覆盖现状未展开；
- `xstars-dev` 的 `build_installer.py` 与 `mac_porting_notes.md` 内部行号存在 ±5 漂移（本报告凡标〔C〕者需 Plan 阶段二次核对）；
- `pyproject.toml` 是否真无 `[project.scripts]` 由 lane B 报告、父 Agent 未逐行复核。

---

## 7. Persisted report

- 本报告路径：`/Users/frank/Documents/GitHub/xstars/plans/explore-20260906-macos-installer.md`
- 可直接作为 `/feature-plan plans/explore-20260906-macos-installer.md` 的输入。
- 子代理原始证据（如需回溯全量清点）：
  - `~/.pi/agent/sessions/--Users-frank-Documents-GitHub-xstars--/subagent-artifacts/6e749efb-f518-4d17-98bb-5a0ad4423e21_codebase-cartographer_output.md`（旧资产逐项完成度 + 签名前置）
  - `~/.pi/agent/sessions/--Users-frank-Documents-GitHub-xstars--/subagent-artifacts/dbb2c5bc-58d2-4b92-a60d-7d7634f2faaf_codebase-cartographer_output.md`（两仓库分叉 + 过时项判定）
- 本轮代码库改动：**零**。`git status` 除本报告外应无变更。
