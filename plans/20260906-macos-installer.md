# Plan: XSTARS macOS 独立安装器（.pkg，arm64-only，用户级免提权，未签名）

- **状态：实施中（M0 进行中：T0.1 已完成入库，T0.2/T0.3 待用户交付 xlsm 后收尾）**
- **日期：2026-09-06**（rev 4 更新：2026-09-08）
- **rev：14**
- **基准：main @ `ab30702`**（锚点核对于 2026-09-06）
- **输入**：`plans/explore-20260906-macos-installer.md`（explore 报告）、researcher 外部调研（运行时方案对比 / xlwings 配置机制 / Sequoia 未签名 pkg 行为 / TCC 边界 / V-01~V-05）、codebase-cartographer 本地上下文报告、2026-09-06 用户访谈（11 项裁定）

| Changelog | 日期 | 变更摘要 | 依据 |
| --- | --- | --- | --- |
| rev 1 | 2026-09-06 | 初稿：新建 Plan（路径 A），9 段结构，M0–M6 共 7 个 Milestone，20 个 To-do 任务，17 个功能缺口 | explore-20260906-macos-installer.md + researcher 调研 + cartographer 报告 + 2026-09-06 访谈 11 项裁定 |
| rev 2 | 2026-09-06 | 父 Agent 独立审计后的事实校正（不涉及任何产品/范围决策）：① 补入「旧仓锚点 ground-truth 校正表」，消除正文声明的 ±5 行漂移；② To-do 任务实测为 **23** 个（rev 1 摘要误记 20），四要素齐备、与 17 个缺口双向映射无孤立；③ 校正 `application/export.py` 的 Pillow 证据行号 | `grep -n` 实测 xstars-dev 与当前仓库；Plan 结构审计脚本输出 |
| rev 3 | 2026-09-06 | **用户批准 Plan（rev 2）并裁定三项**：① 行尾修正采用方案 A（`.gitattributes` 追加 `*.bas text eol=lf`，否决方案 B）；② 推送 `feat/macos-installer` 分支供 Windows 侧 M0 制作；③ 进入实施。新增 R24 与待决 D11（VBE CRLF 残余风险的预登记回退路径）；修正 R21 与 §9 中 `.gitattributes` 的处置描述 | 2026-09-06 用户明确批准 + 三项裁定；`git ls-files --eol` 实证 |
| rev 4 | 2026-09-08 | **T0.1 完成 + 待决 D2 裁定**：① D2 裁定为「内置兜底」——xlam 内置隐藏 `xlwings.conf` sheet，`Interpreter` 留空（researcher 双保险方案）；② `XSTARS.xlam` 在 Windows 侧由 Agent 全自动制作并入库（`installer/mac/assets/XSTARS.xlam`，commit `20ba6c8`）：customUI14→2006 转换（去 1 处 `insertAfterMso`）、真 Excel 16.0 COM 导入 `RibbonCallbacks`、OOXML 注入 customUI part；验收断言全 PASS（2006 命名空间、无 customUI14/insertAfterMso、真机打开无修复、模块与 `.bas` 逐行一致〔仅行尾归一化，D11 范围〕、`xlwings.conf` VeryHidden）。③ **实施环境事实**：本机 WPS Office 在 HKCU 劫持 Excel CLSID（`{00024500-...}` → `et.exe`），COM 自动化需启动 Office16 `EXCEL.EXE /automation` 后经 ROT 绑定真实例；Excel COM `VBProject` 为 null 时优先排查此劫持 | 2026-09-08 用户访谈裁定 D2 + Agent 实施记录（commit `20ba6c8`） |
| rev 5 | 2026-09-08 | **D11 裁定落盘**：T0.1 逐字节断言改为「行尾归一化（`\r\n`→`\n`）后逐字节一致」。实测证据：oletools 提取制品内 RibbonCallbacks 源码 5452B 全 CRLF（163 行）vs 仓库 `.bas` 5289B 全 LF，差值恰为行数→纯 VBE 存储行为；归一化后逐字节一致，两侧均含 `Attribute VB_Name` 头，**制品无需重做**。另补入 F 项验证：customUI rel 包根挂载（2006 type）与用户 Mac 实测渲染中的 `xlwings/addin/xlwings.xlam` 逐模式一致；8.1 安装器单测覆盖面同步扩充 | 2026-09-08 用户确认 D11 回退；父 Agent 独立复验（oletools 提取 + `git ls-files --eol`） |
| rev 6 | 2026-09-08 | **M0 闭合**：① T0.2 完成——`XSTARS_mac.xlsm` 源自 DC2 验收工作簿（无需 Windows 折返），三模块归一化后与仓库 `.bas`/0.37.0 wheel 逐字节一致、`Dictionary` 在位、`xlwings.conf!Interpreter` 已清空，入库（`b9fac14`）；② **新增隐私发现与清洗**：Excel 2010+ 保存时在 `xl/workbook.xml` 写入 `x15ac:absPath`（泄露构建机用户路径 `/Users/frank`、`C:\Users\<user>`），两制品均外科手术移除（xlam 为 `9b92364`），`vbaProject.bin` 逐字节不变，T2.2 复扫范围增补 absPath；③ T0.3 完成（assets/README.md：用途矩阵 + 5 条硬不变量 + 双主机再制作步骤 + WPS CLSID 劫持/Mac VBE 拒 `.cls` 绕行）；④ D3/D5 随交付物裁定闭合。**M0 三任务全部完成，M1 解锁** | 用户交付另存文件 + 父 Agent zipfile/oletools 独立复验与清洗 |
| rev 7 | 2026-09-08 | **M1 完成（T1.1–T1.4 [x]，`aef6a32`）**：① D1 落定——pin `20260901`/`cpython-3.12.14` aarch64 `install_only_stripped`，SHA256 `81a359f1…e4b2b`、size 24,981,445（官方 SHA256SUMS 双源核对）；② `build_pkg.py`（纯函数+注入式下载/命令执行器）+ `runtime.lock.json` + 15 项无网络单测全绿；③ 真实干跑通过：staging import（脱离仓库根）`xstars.__file__` 位于 staging site-packages、依赖装齐、无 dev 项、`xlwings.applescript` 已汇集；④ 新增仓库级事实与处置：ruff 全仓存量 117 error（`xstars/main.py` 34 等，main 从未 ruff-clean 且相关文件在「明确不修改」清单）→ ruff 验收范围明确为**新文件 0 error**；staging 构建产物目录入 `.gitignore`（与 `installer/output/` 同类）并在 compileall/ruff 中排除；⑤ VBA 占位符澄清：断言用 `/Users/(?!<User>)`——上游 `xlwings.bas` 规范占位 `/Users/<User>` 不是泄漏（worker 裁决，D11 语义补充）；⑥ 实施记录：worker 45min 超时一次（pip 装栈耗时），编排者接手完成验证与收尾；oletools 入 `[dev]`（T1.4 授权决定）。全量回归 374 passed/13 skipped 零新增失败；ribbon 门禁空 diff | pytest/ruff/compileall/真实干跑实测；worker 报告 + 编排者复验 |
| rev 8 | 2026-09-08 | **M2 完成（T2.1–T2.4 [x]，`f352a05`）**：真实出包 `XSTARS-1.1.1.pkg`（189,756,535 B，unsigned 预期，SHA256 `7b8d128d…`）；**RK-08 实测解除**——`install-location /` + payload 相对布局 + `installer -target CurrentUserHomeDirectory` 实际落位 `~/Library/Application Support/XSTARS/`（测试安装+receipt 已完整清理）；`distribution.xml` 四项改造落地（arm64/min12.0/currentUserHome/**移除 rootVolumeOnly**）；安装器单测 23/23；全量回归 382 passed/13 skipped。**新增残余风险（已裁决接受）**：xlwings wheel 内置 `quickstart.xlsm` 含上游 `x15ac:absPath`（`C:\Users\felix\…`），系第三方文件且用户不会打开，不做包级清洗以维持“运行时=纯净 wheel 安装”可复现性；T2.2 扫描范围维持两件 XSTARS 制品 | worker 报告 + 编排者独立复验（pytest/ruff/compileall/门禁/单测重跑于 pi-lens 重排后） |
| rev 9 | 2026-09-08 | **M3 完成（T3.1–T3.3 [x]，`15536c1`）**：postinstall.sh（201 行，幂等：payload 解包 fail-closed + xlam→Startup 预创建 + applescript→Application Scripts + INTERPRETER_MAC 写 Containers conf 保留未知键；无任何 rm -rf、不碰 `~/.xstars`）；`render_xlwings_conf()` 纯函数（合并逻辑单测）；**新发现并处置**：pkgbuild 会把 Sequoia provenance xattr 转为 `Scripts/._postinstall` AppleDouble → fail-closed 清理流程（expand→剥→flatten→重验，最终 pkg 零 AppleDouble，真实重出包 189,759,482 B，Scripts 含 postinstall+xlwings.conf）。真实安装未执行（V-02/V-03 责任人=用户，边界遵守）。安装器单测 28/28；全量回归 387 passed/13 skipped | worker 报告 + 编排者独立复验（含 postinstall 安全审计：无 rm -rf/无 ~/.xstars 触碰/无路径泄漏） |
| rev 10 | 2026-09-08 | **M4 完成（T4.1–T4.2，详见提交）+ T2.2 验证深度缺陷更正**：① uninstall.sh（250 行，默认 dry-run/`--apply` 真删，六步清理含 RegistrationDB `OPENn` 备份+integrity_check 失败即回滚；Excel 运行中拒绝执行；不碰 `~/.xstars`）；② docs/macos-installer.md 卸载章节先行（安装/放行属 T5.2）；③ **T2.2 验证深度缺陷更正（worker 发现）**：M3 的零 AppleDouble 仅查 pkgutil 外层，内层 component Payload cpio 实有 `._XSTARS-payload.tar.gz` 等 4 条 → build_pkg.py 加内层 cpio 清洗，重出包后内层外层均为 0，嵌套断言入 tests；8.2 表同步扩充。安装器单测 31/31；全量回归 390 passed/13 skipped | worker 报告 + 编排者独立复验（终态重出包 + cpio 实测 + uninstall 安全审计） |
| rev 11 | 2026-09-08 | **M5 完成（T5.1–T5.5 [x]，`c52f80c`）**：① 版本单源——`xstars/__init__.py` 1.0.0→1.1.1 + 防漂移单测（tomllib，CI 3.10 回归用 tomli 条件导入）；② pyproject 显式声明 `Pillow>=10.0`（直呼导入点 main.py:1053/export.py:328）；③ `docs/macos-installer.md` 补全（含 M2 实测的 CurrentUserHomeDirectory 落位语义、Sequoia 三途放行、无右键打开过时话术）；④ 双语 README 安装包模式为推荐路径、开发者模式降级 fallback；⑤ 三 docs 同步（manual-acceptance 增安装器验收组；strategy 路线表「独立 `.pkg` 安装器交付中」）；⑥ ribbon/README ship 制品澄清。安装器单测 32/32；全量回归 391 passed/13 skipped；过时话术 grep 零命中；ribbon 门禁空 diff。**新事实**：CI 既有测试 job 跑 Python 3.10，新测试已兼容（tomli 条件导入）；tests/ 存量 ruff 21 error（M4 HEAD 同，非本次引入，不在范围） | worker 报告 + 编排者独立复验（含策略/过时话术 grep 与 T5.1 逐行 diff） |
| rev 12 | 2026-09-08 | **T6.1 完成（`10cab2d`）**：`.github/workflows/macos-support.yml` 新增 `macos-pkg-build` job（macos-latest 已为 arm64，经 runner-images 元数据核实；Python 3.12 构建宿主；**两步构建** `--prepare-runtime` → `--build-pkg`——编排者任务规格曾误写单命令，worker 预检拦下并裁决修正；installer 测试 + compileall + upload-artifact retention 7d；permissions contents: read，无 secrets）；既有三 job 未动；YAML 有效（psych）。M6 标记进行中：T6.2（真机 V-01~V-05）待用户，随 PR 首次 CI 运行验证 job 本身 | worker 报告 + 编排者复验（YAML psych + diff 全文审读） |
| rev 13 | 2026-09-08 | **Review 第 1 轮（三路 fresh-context）完成 + F1–F10 修复批落地**：coverage lane 产 14 findings（1 Blocker + 4 值得修复 + 6 可选 + 3 延期）；correctness lane 产 1 Blocker + 2 值得修复（fallback 模型续跑成功）；quality lane 产 3 值得修复（fallback 模型续跑成功）；另两 lane 首跑因 provider 500 中断后恢复。修复批（fix worker，全验收绿：安装器单测 43/43、全量 402 passed/13 skipped、pkg 重出 ≈137MiB、三层 AppleDouble 零）详见文末「Review 记录」；可选改进 9 项记入残余清单不在本轮实施；RK-01 处置：staging 实测 Tcl/Tk 资源在位 + F1 的 CI 冒烟 import ttkbootstrap 间接覆盖 Tk 链路，sitecustomize 缓解推迟至 V-01 实测需要时 | 三路 reviewer 报告 + fix worker 报告 + 编排者复验（43/43、402/13、ruff、YAML、bash -n、门禁） |
| rev 14 | 2026-09-08 | **R9 修正（用户访谈裁定）+ 第 2 轮 A–D 修复闭合**：真实 `--prepare-runtime` 解析出 numpy 2.5.3 / scipy 1.18.1（wheel 标签 `macosx_14_0_arm64`，要求 macOS 14+），原 R9=12.0 成为假承诺 → **最低 macOS 修正为 14.0（Sonoma），不钉版本**；落地：distribution.xml `min="14.0"`、测试断言、双语 README、docs 同步（grep 无残留）。第 2 轮复审 A–D 全闭合：A 冒烟显式导入全部 11 项运行时依赖 + `sys.prefix` 来源断言 + `pip check`；B 卸载三态（备份→恢复 / marker→删除 / 皆无→警告保持）；C manifest 绑定 staging 内容（三文件 SHA256 + staging 解释器实跑版本 + 工作簿内嵌 xlwings 版本比对，篡改即 fail-closed）；D 卸载文档三路径语义。验证：安装器单测 45 passed/1 skipped；全量 404 passed/14 skipped | 用户访谈裁定（R9 修正）；第 2 轮 reviewer 报告 + fix worker 报告 + 编排者复验 |

> 锚点标注约定：本文引用的 **当前仓库（xstars）** 锚点已由 feature-planner 于 2026-09-06 在 main @ `ab30702` 上二次实读核验（含 explore 报告中标〔C〕者）。**旧仓库（xstars-dev）** 锚点来自 explore 报告〔P〕实测与 cartographer 报告〔C〕逐行复核，两份报告对同一锚点存在 ±5 行漂移（如 `build_pkg.sh` 签名触发段：explore 标 `:97-100`〔P〕，cartographer 标 `:87-94`〔C〕），移植时以旧仓库实际文件为准，**凡引用 xstars-dev 行号处实施前需打开原文件二次核对**。

### 旧仓锚点 ground-truth 校正表（rev 2，父 Agent 2026-09-06 `grep -n` 实测）

> 本表为权威值。正文与 explore 报告中凡与本表冲突的 xstars-dev 行号，**以本表为准**；实施时仍应打开原文件确认语义。

| 旧仓文件 | 正文/explore 曾引用 | **实测真实行号** |
| --- | --- | --- |
| `installer/mac/build_pkg.sh` | `VERSION=` 强制传入 `:10` | **`:11`**（`ROOT=` 在 `:12`） |
| 同上 | staging `WORK=$(mktemp -d)` + trap `:19-20` | `:20-21` 区间（trap 清理） |
| 同上 | `PAYLOAD`/`PAYLOAD_RES` `:36-38` | **`:40-41`**（`PAYLOAD_RES="$PAYLOAD/private/var/tmp/xstars-install"`） |
| 同上 | `COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata` `:47-48` | 说明注释 **`:48`**，命令本体 **`:51`** |
| 同上 | `osacompile` `:56-57` | **`:64`** |
| 同上 | `SCRIPTS` staging + `cp postinstall` `:58-61` | **`:67`** 起 |
| 同上 | `pkgbuild` `:64-70` | **`:74-79`**（`--identifier` `:76`、`--install-location "/"` `:78`、`--scripts` `:79`） |
| 同上 | `FINAL_PKG=` + `productbuild` `:72-81` | **`:86-88`**（`--distribution` `:88`） |
| 同上 | 签名触发段 `:87-94`〔C〕 | **`:97-102`**（explore 的 `:97-100`〔P〕正确；`:99` 传 `XSTARS_PKG`/`XSTARS_VERSION`，`:102` 为不可执行告警） |
| `installer/mac/sign_and_notarize.sh` | 无条件回跑 build_pkg `:61` | **`:61`** ✓ 正确（递归缺陷成立） |
| `installer/mac/distribution.xml` | `hostArchitectures` `:11`、`os-version min` `:14` | **`:11`** / **`:14`** ✓ 均正确 |
| `installer/mac/postinstall.sh` | `set -e` `:11` | **`:15`** |
| 同上 | tarball 缺失检查 + `exit 1` `:17-20` | **`:21-23`**（`exit 1` 在 `:23`） |
| 同上 | `stat -f '%Su' /dev/console` `:41` | **`:43`** |
| 同上 | `dscl ... NFSHomeDirectory` `:43-44` | **`:45`** |
| 同上 | `chown` xlam `:49` | **`:51`** |
| 同上 | `sudo -u "$ACTIVE_USER" mkdir -p` `:67` | **`:68`** |
| 同上 | `chown` scpt `:69` | **`:70`** |
| 同上 | 末尾 `exit 0` `:80` | **`:80`** ✓ 正确 |
| `installer/build_installer.py` | `_extract_version()` `:49-55` | **`:55-60`**（读 `__init__.py` **`:57`**、正则 **`:58`**、失败 raise `:60`） |
| 同上 | PyInstaller 调用 `:81-84` | **`:90`** |
| 同上 | `ensure_xlam_for_mac` 的 `XLAM_PATH.exists()` `:166` | **`:167`** |
| 同上 | 输出命名 `installer/output/XSTARS-{version}.pkg` `:302` | **`:308`** |
| 同上 | `--skip-*` 参数 `:317-322` | **`:320`** 起（`--target` 亦在 `:320`） |

**当前仓库一处证据精度校正**：`Pillow` 的直接使用点实测为 `xstars/main.py:1053`（`from PIL import ImageGrab`）与 `xstars/application/export.py:328`（`from PIL import ImageGrab`）；`export.py:340` 实为 `raise ContractError(ErrorCode.EXPORT_DPI, ...)`，**非 PIL 调用**。凡以 `export.py:340` 作为 Pillow 证据处，应改以 `:328` 为准（R/Pillow 相关任务与 Gap 的结论不受影响：Pillow 确为直接使用却未在 `pyproject.toml:12-20` 声明）。

---

## 1. Goal

在当前 main 架构（xlwings `RunPython` 同进程模型、`ribbon/*.bas` 零修改）之上，交付一条可重复的 macOS `.pkg` 独立安装器链路：

- **终端用户价值（可验证结果）**：Apple Silicon Mac 用户双击 `.pkg`（或一条 `installer` 命令）装完 XSTARS——全程**不安装 Python、不建 venv、不手工导入任何 VBA**；打开 Excel 即见 XSTARS 功能区，用预置 `XSTARS_mac.xlsm` 模板即可跑通 Run（统计）与 Export（图片导出）两类核心操作；卸载有官方脚本，卸干净后 Excel 启动不再弹「找不到加载项」。
- **开发者价值**：`installer/mac/build_pkg.py` 一条命令出包（或 CI 出包），装配逻辑 100% Python 可测，续接仓库既有 pytest 门禁；不触碰任何 VBA、不破任何既有 CI 硬门禁。
- **明确不做**：签名/公证、universal2、`.app`/DMG/Homebrew、PyInstaller（详见 Non-goals）。

---

## 2. Requirements

> 编号 R1–R11 逐条对应 2026-09-06 访谈的 11 项用户裁定；R12–R23 为前序访谈锁定的方案骨架与代码库硬约束。标注〔硬〕者为不可妥协约束。

| ID | 需求 | 来源 | 约束级别 |
| --- | --- | --- | --- |
| R1 | **运行时来源**：构建期从 GitHub 下载 `python-build-standalone` 的 `aarch64-apple-darwin` `install_only_stripped` 归档，**固定 SHA256 校验**，校验失败即终止构建（fail-closed）；运行时 tarball **不入库**。 | 2026-09-06 访谈裁定 1 | 〔硬〕 |
| R2 | **内置 Python 版本 3.12**（cpython-3.12.x，具体小版本与钉值见待决事项 D1）。 | 访谈裁定 2 | 〔硬〕 |
| R3 | **用户级免提权安装**：安装位置 `~/Library/Application Support/XSTARS/`；`distribution.xml` 使用 `domains enable_currentUserHome="true"`；**不写 /Applications、不要求管理员密码**（与 `docs/wps-installation.md:34` 单用户免提权约定一致，见已核对该文件 §3「安装器为当前用户单用户安装（无需管理员提权）」）。 | 访谈裁定 3 | 〔硬〕 |
| R4 | **入库二进制制品范围**：`XSTARS.xlam`（2006-format customUI ribbon）**+** 预置 `XSTARS_mac.xlsm` 模板（工作簿内嵌 `RibbonCallbacks` + `xlwings.bas` + `Dictionary` 三模块）。依据：`ribbon/README.md:67`（实测确认功能区回调在活动工作簿解析，该行已核对，原文 "Excel resolves `RibbonCallbacks.*` in the **active workbook**"）；`docs/macos-developer-setup.md:88-99`（§4 要求工作簿内嵌三模块，已核对）。 | 访谈裁定 4 | 〔硬〕 |
| R5 | **制品制作人/时点**：**用户在实施开始前手工制作并交给 Agent 入库**——这是 M0 的输入前提，Plan 中显式标为**阻塞前置**；制作步骤引用 `ribbon/README.md:44-51`（RibbonX Editor 打包 2006-format xlam 的已验证流程，已核对）与 `docs/macos-developer-setup.md:88-99`。**rev 4 实施记录**：T0.1 已例外地由 Agent 在 Windows 侧全自动完成（真 Excel 16 COM + OOXML 注入，绕过 WPS CLSID 劫持，见 Changelog rev 4）；T0.2（xlsm）仍为用户手工（Mac Excel 无 VBProject，不可自动化） | 访谈裁定 5；rev 4 实施记录 | 〔硬，阻塞前置：T0.2 仍阻塞〕 |
| R6 | **版本单一来源**：以 `pyproject.toml:7`（已核实 `version = "1.1.1"`）为唯一来源，构建脚本读 pyproject（`tomllib`，**不再**像旧 `build_installer.py:49-55`〔C〕那样正则抓 `__init__.py`）；并把 `xstars/__init__.py:3`（已核实 `"1.0.0"`）对齐到 `"1.1.1"`。 | 访谈裁定 6 | 〔硬〕 |
| R7 | **卸载**：交付 `uninstall.sh` + 文档；必须覆盖 Excel 启动项清理与 `~/Library/Group Containers/UBF8T346G9.Office/MicrosoftRegistrationDB/*.reg` 中 `OPENn` 残留清理（依据 `docs/macos-developer-setup.md:191-193`「cannot find add-in」故障段，已核对）；必须含**备份步骤**与 `PRAGMA integrity_check`。 | 访谈裁定 7 | 〔硬〕 |
| R8 | **CI**：新增 `macos-latest` build job，产出 `.pkg` 并上传为 workflow artifact（可参照既有 `.github/workflows/macos-support.yml:16` 的 macOS job，已核对）。 | 访谈裁定 8 | 硬 |
| R9 | **最低 macOS 14.0（Sonoma）**（rev 14 修订：真实构建解析出 numpy 2.5.3 / scipy 1.18.1 的 `macosx_14_0_arm64` wheel，原 12.0 承诺不成立；用户裁定不钉版本、随栈上浮）；`distribution.xml` 的 `<os-version min>` 为 14.0。 | 访谈裁定 9；rev 14 用户访谈修订 | 〔硬〕 |
| R10 | **顺手修（仅此一项纳入）**：`pyproject.toml` 显式声明 `Pillow`（现靠 matplotlib 间接引入，但 `xstars/main.py:1053`、`xstars/application/export.py:328,340` 直接使用 `from PIL import ImageGrab` / `Image`——两处均已实读核对）。 | 访谈裁定 10 | 硬 |
| R11 | **PR 边界**：单分支 `feat/macos-installer` + **单个 Draft PR**，所有 Milestone 串行进入同一 PR。 | 访谈裁定 11 | 〔硬〕 |
| R12 | **`ribbon/*.bas` 零修改**：CI 硬门禁 `.github/workflows/macos-support.yml:65-73` job「Existing VBA files are unchanged」→ `git diff --exit-code origin/main...HEAD -- 'ribbon/*.bas'`（已实读核对 `:73`）。任何触碰即红。 | 方案骨架；explore §3.7 | 〔硬〕 |
| R13 | **保留 xlwings `RunPython` 同进程模型**：不新建 Mac 专用 VBA 唤起分支，不复活旧版「冻结 exe + VBA Shell 子进程」模型。 | 方案骨架；explore §1.4/§1.5 | 〔硬〕 |
| R14 | **不做签名、不做公证**：产物为未签名 `.pkg`；**不移植 `sign_and_notarize.sh`**（避开旧 `build_pkg.sh:87-94`〔C〕/`:97-100`〔P〕触发 + `sign_and_notarize.sh:61`〔P〕无条件回跑的无限递归缺陷），构建流水线为单向。 | 方案骨架；explore §2.1 | 〔硬〕 |
| R15 | **arm64-only**：不做 universal2；`distribution.xml` 的 `hostArchitectures` 由旧版 `arm64,x86_64` 改为 `arm64`。 | 方案骨架；旧 `distribution.xml:11`〔P〕 | 〔硬〕 |
| R16 | **安装器投递/配置四件事**：① 内置 Python 3.12 运行时 + 已装依赖的 site-packages（含 xstars 本体，非 editable）→ `~/Library/Application Support/XSTARS/python/`；② `XSTARS.xlam` → `~/Library/Group Containers/UBF8T346G9.Office/User Content.localized/Startup.localized/Excel/`；③ `xlwings.applescript` → `~/Library/Application Scripts/com.microsoft.Excel/`（直接投递文件，无需调用 `xlwings runpython install`）；④ 写 `INTERPRETER_MAC` 到 `~/Library/Containers/com.microsoft.Excel/Data/xlwings.conf`（键名大写 `INTERPRETER_MAC`，支持 `$APPLICATIONS`/`$HOME` 变量）。 | 方案骨架；researcher §B | 〔硬〕 |
| R17 | **不使用 PyInstaller、不产出 `.app`**：xlwings 官方文档明确 `RunFrozenPython`「Currently only available on Windows」；`RunPython` 需要真实解释器可执行文件。 | 方案骨架；researcher §A.5 | 〔硬〕 |
| R18 | **`xlwings addin install` 不需要**：避免官方 `xlwings.xlam` 与 `XSTARS.xlam` 争功能区。 | 方案骨架；researcher §B.3 | 硬 |
| R19 | **Sequoia `com.apple.provenance` → AppleDouble 陷阱**：staging 阶段沿用旧 `build_pkg.sh:47-48`〔C〕的 `COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata` 手法；构建脚本另需避免向 payload 引入任何 `._*` 伴生文件，并加产物断言。 | 方案骨架；explore §1.2 | 〔硬〕 |
| R20 | **可测试面**：装配逻辑写成可测 Python（`installer/mac/build_pkg.py` 抽纯函数），bash（postinstall/uninstall）只做薄脚本；darwin 替身与平台注入沿用 `tests/test_macos_support.py:43-104`（`StrictPicture`/`DarwinApp`/`_darwin_book`/`_install_tk_modules`）与 `patch("xstars.main.sys.platform", "darwin")` 约定。仓库 tests/ 为 100% pytest、无 shell 测试框架先例。 | 方案骨架；cartographer §3 | 硬 |
| R21 | **制品治理**：制品放 `installer/mac/assets/` 下不会被 `.gitignore:40`（精确路径 `installer/XSTARS.xlam`，已实读核对）命中；模板**必须**命名 `.xlsm`（`.gitignore:10` 为 `*.xlsx`，已核对）；无 LFS（`.gitattributes` 原仅 `* text=auto`，**rev 3 已追加 `*.bas text eol=lf`（R24）**，仅影响 `.bas` 行尾；OOXML/GIF 经 `git ls-files --eol` 实测为 `-text` 不受影响），制品以常规 blob 入库（与 `XSTARS_Templates.xlsx`、`assets/*.gif` 现状一致）。 | 方案骨架；cartographer §2（已二次核对 .gitignore/.gitattributes）；rev 3 用户裁定 A | 硬 |
| R22 | **文档同步**：7 处声明「macOS 仅开发者模式 / 不提供安装器」的文档段落须更新（清单见 §9 文件级范围）。 | cartographer §5 | 硬 |
| R23 | **真机验收**：V-01~V-05 五项真机验证由**用户在 Mac mini 执行**并回填 Draft PR（沿用 `plans/20260829-macos-support.md:18` 门禁与 `docs/macos-manual-acceptance.md:3-5`「CI 不启动 Excel」约定）；Agent 不执行真实 Excel/macOS 宿主冒烟。 | 方案骨架 | 〔硬〕 |
| R24 | **`.bas` 行尾钉死 LF（rev 3 新增）**：`.gitattributes` 追加 `*.bas text eol=lf`，保证 Windows 检出的 `ribbon_callbacks.bas` 与仓库一致（LF），使 T0.1 的逐字节断言在跨机制作下可满足。**残余风险预登记**：Excel VBE 可能在 `vbaProject.bin` 内以 CRLF 存储模块源码（与输入行尾无关）——若 T0.1 逐字节断言失败，按 D11 回退为断言侧行尾规范化（原方案 B），届时**制品无需重做**。 | 2026-09-06 用户裁定（选 A，否 B） | 〔硬〕 |

---

## 3. Non-goals

以下各项本轮**明确不做**（用户未选中或已明确排除）：

1. **不签名、不公证**：不申请 Developer ID、不移植 `sign_and_notarize.sh`、不配 CI 签名 secrets；产物为未签名 `.pkg`（本机构建可直接安装，跨机分发需用户手动放行，文档已覆盖）。
2. **不做 universal2**：arm64-only；Intel Mac 用户不在本轮支持范围。
3. **不产出 `.app` / DMG / Homebrew tap**。
4. **不使用 PyInstaller**（且与 R12/R17 互斥，事定排除）。
5. **不修改 `ribbon/*.bas` 任何文件**（`ribbon_callbacks.bas`、`ribbon_callbacks_installed.bas`、`ribbon_callbacks_standalone.bas` 均零改动，CI 门禁见 R12）。
6. **不在 macOS 上自动生成 `.xlam`**：物理不可行（Excel for Mac 无 `VBProject` AppleScript 能力，researcher §B.4；`docs/macos-developer-setup.md:33` 与 `ribbon/README.md:44-51` 均要求 Windows/RibbonX Editor 侧打包）。
7. **不支持 WPS for Mac**。
8. **不修 `requirements.txt` 陈旧子集**（缺 ttkbootstrap；CI 已不使用该文件，`macos-support.yml:24-27,47-50` 均 editable 安装）。
9. **不恢复 `xstars/cli.py:56` 的未保存工作簿回退**（属 fail-closed 设计取舍，已有测试锁定，见 explore §1.5）。
10. **不清理 `installer/wps/.ruff_cache` 等缓存目录**。
11. **不把内置运行时 tarball 入库**（构建期下载 + SHA256 校验）。
12. **不由 Agent 执行真实 Excel/macOS 宿主冒烟**（沿用既有门禁，真机项由用户执行并回填 Draft PR，R23）。

---

## 4. Research summary

### 4.1 代码库现状（含两代架构对比结论）

- **当前仓库无任何 macOS 安装器**：`installer/` 下仅 `wps/`（Windows 侧三件套：`.spec` + `.iss` + `build.ps1`）；全仓库 `*.xlam` 与 `*.entitlements` 均为空（explore §2.2〔P〕）。当前文档契约明确「非独立应用、无 .app/签名/公证」（`docs/macos-developer-setup.md:3,5`〔P〕，已核对）。
- **两代架构对比结论**（explore §1.4/§1.5〔P〕）：旧版 = 扁平两层 + 冻结 exe 子进程 + pickle 跨进程缓存 + Python 侧**零**平台分支（`darwin`/`sys.platform` 0 处）；新版 = 三层（共享内核 / `application/` 宣主无关层 / 宣主适配层）+ 三宿主三进程 + `artifacts.py` JSON 工件（`SCHEMA_VERSION`、`_SAFE_KEY = ^[0-9a-f]{64}$`、原子写）+ `darwin` 仅 8 处收敛在 `main.py`（已核对 `xstars/artifacts.py:46-49`）。**判定：当前架构明显更优**（宣主解耦、平台差异可测性 tests 61 处 `darwin` vs 旧 0、pickle→JSON 安全升级、复杂度有归处）。**对本轮的指导意义**：只移植旧版唯一领先的部分（能出包的 `.pkg` 链路），**不复活它的进程模型**；选定方案（自带非冻结 Python）是当前架构的原生形态——业务代码一行不改，动态 import（`main.py:26,27` 等〔P〕）在真解释器 + 真 site-packages 下天然正常。
- **版本双源冲突**（已实读核对）：`pyproject.toml:7` = `1.1.1` vs `xstars/__init__.py:3` = `1.0.0`；旧 `build_installer.py:51`〔C〕从 `__init__.py` 取版本 → 直接移植会产出 1.0.0 的包。裁定 R6：pyproject 为单源。
- **Pillow 未显式声明**（已实读核对）：`pyproject.toml` 无 `Pillow`；但 `xstars/main.py:1053`（`from PIL import ImageGrab`，Windows 高分辨率导出路径）与 `xstars/application/export.py:328,340`（WPS 剪贴板导出 `ImageGrab`/`Image`）直接使用。裁定 R10。
- **CI 现状**（已实读核对 `.github/workflows/macos-support.yml` 全文）：三 job（macos/windows 测试 + ubuntu VBA 零差异门禁 `:65-73`）；无签名 job、无 artifacts 发布、无证书 secrets。安装依赖全部 `pip install -e ".[dev]"`；`requirements.txt` 未被 CI 使用且缺 `ttkbootstrap`（陈旧子集，不修，Non-goal 8）。
- **tests/ 组织与可测试面**（cartographer §3）：100% pytest、无 shell 测试先例；`tests/test_macos_support.py:43-104` 提供 `StrictPicture`/`DarwinApp`/`_darwin_book`/`_install_tk_modules` 替身与 `patch("xstars.main.sys.platform", "darwin")` 平台注入约定；`skipif` 仅用于无法 mock 隔离的 C 级 Tk/COM 场景。**结论**：装配逻辑 Python 化 + pytest 单测（R20）。
- **旧脚本逐行可移植点**（xstars-dev，cartographer §4〔C〕逐行复核；行号有漂移，实施前二次核对）：
  - `build_pkg.sh`：`XSTARS_VERSION` 环境变量强制传入（`:10`）；staging 布局 `WORK=$(mktemp -d)` + trap 清理（`:19-20`）；`COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata` 规避 AppleDouble（`:47-48`）；pkgbuild `--identifier com.frank-sysu.xstars --install-location /`（`:64-70`）；productbuild + `sed s/__VERSION__/`（`:72-81`）；签名递归缺陷（`:87-94`〔C〕/`:97-100`〔P〕触发 → `sign_and_notarize.sh:61`〔P〕回跑 → 死循环）。
  - `postinstall.sh`：`stat -f '%Su' /dev/console` 探测控制台用户（`:41`）、`dscl` 查真实 Home（`:43-44`）、`sudo -u` + `chown` 属主归还（`:49,67,69`）、`set -e` + 关键 payload 缺失才致命（`:17-20`）、用户级拷贝失败降级为指引输出而非致命（`:53-58`）、末尾显式 `exit 0`（`:80`）。
  - `distribution.xml`：`hostArchitectures="arm64,x86_64"`（`:11`〔P〕，本轮改 `arm64`）；`<os-version min="11.0"/>`（`:14`〔P〕，本轮改 12.0）；`domains enable_localSystem="true"`（本轮按 R3 改为 `enable_currentUserHome="true"`）；`__VERSION__` 占位符。
- **制品治理**（cartographer §2，已二次核对 `.gitignore`/`.gitattributes` 原文）：`installer/mac/assets/XSTARS.xlam` 不命中 `.gitignore:40`（该行为精确路径 `installer/XSTARS.xlam`）；`.xlsm` 不被 `.gitignore:10`（`*.xlsx`）忽略；无 LFS；与 `XSTARS_Templates.xlsx`、`assets/*.gif` 的常规 blob 现状一致（R21）。

### 4.2 现有模式与约定（沿用）

- **`installer/<product>/` 三件套**：目录 + 构建脚本 + 打包规格，按产品线分目录（`installer/wps/` 先例），不共享跨产品安装包。
- **版本构建期注入**：WPS 侧 `XSTARS_WPS.iss:4-5` `#ifndef` 兜底 + `build.ps1:85-94` 构建期读 `package.json`〔P〕→ 本轮改为读 `pyproject.toml`（R6）。
- **fail-closed 优于回退**：macOS 导出无 artifact 直接报错不截图；构建期 SHA256 校验失败即终止（R1 同哲学）。
- **文档即验收契约**：真机验收由人执行并回填 Draft PR（R23）。

### 4.3 外部方案对比（researcher 报告，结论附链接 + 日期 + 置信度）

| 方案 | 结论 | 关键证据 | 取舍 |
| --- | --- | --- | --- |
| **python-build-standalone（推荐）** | `aarch64-apple-darwin` 自包含、完全可重定位 CPython；官方构建内置动态共享库版 Tcl/Tk（PR #676，2025-08-08，置信度高）；`install_only_stripped.tar.gz` 约 16.5–18.6 MB、解压约 50–75 MB；MPL-2.0/PSF 宽松许可允许打包分发 | [python-build-standalone running docs](https://gregoryszorc.com/docs/python-build-standalone/main/running.html)（2025-05，高）；[PR #676](https://github.com/astral-sh/python-build-standalone/pull/676)（2025-08-08，高）；[PR #421](https://github.com/astral-sh/python-build-standalone/pull/421)（2024-12-19，高） | **采用**（R1/R2）。已知 Quirk：特定路径下可能 `Can't find a usable init.tcl`，可 `TCL_LIBRARY`/`TK_LIBRARY` 显式指向规避（[quirks docs](https://gregoryszorc.com/docs/python-build-standalone/main/quirks.html)，2024-12，高）→ 风险 RK-01 / V-01 |
| uv 管理 Python | uv 本身只拉取 python-build-standalone；**打包机可用 uv 自动化准备环境，但 uv 二进制不进 .pkg** | [uv Python versions docs](https://docs.astral.sh/uv/concepts/python-versions/)（2024-10，高） | 构建/CI 侧可选使用；终端用户零感知 |
| venv 整体搬迁 | **不可行**：PEP 405 `pyvenv.cfg` 的 `home` 为绝对路径，搬迁后标准库定位失败；console script shebang 硬编码 | [PEP 405](https://peps.python.org/pep-0405/)（高）；[CPython #136051](https://github.com/python/cpython/issues/136051)（2024-2025，高） | **排除**。依赖必须装进 standalone 运行时主 site-packages |
| conda-pack / Miniforge | **排除**：arm64 上替换二进制前缀破坏 Mach-O 签名 → 内核 `SIGKILL: 9`；体积 400MB–1GB；Anaconda defaults 许可陷阱 | [Uwe Korn 博客](https://uwekorn.com/2024/03/11/getting-codesigning-to-work-with-apple-silicon.html)（2024-03-11，高）；[conda-pack #126](https://github.com/conda/conda-pack/issues)（中） | **排除** |
| PyInstaller | **排除**：`RunPython` 把代码动态传给通用解释器（`$INTERPRETER -c "..."`），冻结二进制无法充当；xlwings 官方明确 `RunFrozenPython`「Currently only available on Windows」 | [xlwings deployment docs](https://docs.xlwings.org/en/stable/deployment.html)（2024-2025，高） | **排除**（R17） |
| xlwings 配置机制 | 优先级：工作簿 `xlwings.conf` sheet > 同目录 `xlwings.conf` 文件 > 用户全局配置；**macOS 用户全局配置确切路径 `~/Library/Containers/com.microsoft.Excel/Data/xlwings.conf`**（Excel 沙盒内）；键名大写 `INTERPRETER_MAC`（v0.27+）；支持 `$HOME`/`$APPLICATIONS`/`$DOCUMENTS`/`$DESKTOP` 变量展开 | [xlwings addin & settings docs](https://docs.xlwings.org/en/latest/addin.html)（2025，高） | **采用**（R16④）；可选兜底见待决 D2 |
| `xlwings runpython install` | 在 `~/Library/Application Scripts/com.microsoft.Excel/` 投递 `xlwings.applescript`，是 Mac `RunPython` 的硬性前提；**可直接投递文件，无需调用命令**；卸载即删该文件 | [xlwings CLI docs](https://docs.xlwings.org/en/stable/command_line.html)（2024，高） | **采用**（R16③）；`applescript` 文件来源 = 内置运行时 site-packages 内的 xlwings 包数据（**确切文件名实施期核实**，见 4.5 证据缺口） |
| `xlwings addin install` | 仅复制官方 `xlwings.xlam` 到 XLSTART；**非必需**，避免功能区争用 | [xlwings addin docs](https://docs.xlwings.org/en/stable/addin.html)（2024，高） | **不采用**（R18） |
| Sequoia 未签名 pkg 行为 | **Sequoia 已移除右键「打开」绕过**（Apple 官方发行说明确认）；双击被拦后唯一 GUI 路径 =「系统设置 → 隐私与安全性 → 仍要打开」+ 输密码；或 `xattr -d com.apple.quarantine <pkg>` 后双击；或 `sudo installer -pkg <pkg> -target /` 完全免 GUI；本机构建/局域网传 scp/USB 无 quarantine 不受影响；`pkgbuild`/`productbuild` 无证书构建无任何系统阻断 | [Ars Technica](https://arstechnica.com/gadgets/2024/08/macos-15-sequoia-makes-you-jump-through-more-hoops-to-disable-gatekeeper-app-checks/)（2024-08-07，高）；[Michael Tsai](https://mjtsai.com/blog/2024/07/05/sequoia-removes-gatekeeper-contextual-menu-override/)（2024-07-05，高）；[Scripting OS X packaging](https://scriptingosx.com/2025/08/installing-packages/)（2025-08，高） | **分发文档必须按此三途书写**（R7 文档 / V-04） |
| TCC / 权限边界 | 写 `Group Containers/UBF8T346G9.Office`、`Application Scripts/com.microsoft.Excel`、`Containers/com.microsoft.Excel/Data` **均不触发 TCC 弹窗**；但 postinstall 以 root 运行时新文件属主为 root:wheel → 必须探测 Console User（`stat -f '%Su' /dev/console` + `dscl` 查 Home）并 `chown` 归还 | [HackTricks TCC](https://hacktricks.wiki/en/macos-hardening/macos-security-and-privilege-escalation/macos-security-protections/macos-tcc.html)（2025，高）；[Scripting OS X](https://scriptingosx.com/2020/08/running-a-command-as-another-user/)（2020-08，高） | **采用**（postinstall 属主修正逻辑，T3.1）；注：R3 用户域安装下 postinstall 以安装用户身份运行，属主逻辑保留作双保险 |
| Excel for Mac 加载项机制 | Startup 目录 `.xlam` 全自动加载免勾选；`UBF8T346G9.Office` 为受信任组容器；`AppleScriptTask` 硬性限定脚本位于 `~/Library/Application Scripts/<bundle id>/`；Excel for Mac 无 `VBProject`（无法脚本化注入 VBA） | [MacAdmins docs](https://macadminsdoc.readthedocs.io/en/master/Applications/Microsoft-Office-2016.html)（高）；[Microsoft Learn AppleScriptTask](https://learn.microsoft.com/en-us/office/vba/office-mac/applescripttask)（2023-06，高）；[xlwings #2616](https://github.com/xlwings/xlwings/issues/2616)（2024，高） | 印证 R16②③ 与 Non-goal 6 |
| wheel 可用性 | 全部依赖在 arm64 有预编译 wheel（numpy/pandas/matplotlib/Pillow/xlwings = `macosx_11_0_arm64`；scipy = **`macosx_12_0_arm64`**）或纯 Python wheel（seaborn/scikit-posthocs/ttkbootstrap/statannotations）；终端用户机无需 Xcode CLT、无需联网编译 | researcher §A wheel 清单（PyPI，2026，高） | **支撑 R9**（最低 12.0 由 scipy wheel 标签决定） |

### 4.4 推荐方案与替代方案

- **推荐（即本 Plan）**：路径 B1——内置非冻结 Python（python-build-standalone 3.12 arm64）+ 已装依赖 site-packages（含 xstars 非 editable）+ 入库 `XSTARS.xlam`/`XSTARS_mac.xlsm` 制品 + postinstall 四件事投递 + `INTERPRETER_MAC` 指向 `~/Library/Application Support/XSTARS/python/bin/python3` + 未签名 `.pkg`（用户域安装）。
- **替代方案及排除理由**：
  - 路径 A（旧资产直译，冻结 exe + AppleScriptTask 自研 `.scpt`）：必须重写 `installed.bas` 加 `#If Mac` 分支 → 破 R12 硬门禁；复活 pickle 缓存与宣主耦合 → 排除。
  - 路径 B2（PyInstaller `.app`）：与 `RunPython` 互斥（RunFrozenPython 仅 Windows），必然要求改 `.bas` → 排除。
  - 路径 C（签名公证产品化）：本机 0 张有效签名身份〔P〕、无 notary profile〔P〕，为硬阻断；且用户已裁定本轮不签名 → 留待后续。
  - conda-pack / venv 搬迁：见 4.3 表，技术不可行或高风险 → 排除。

### 4.5 证据缺口（显式记录）

1. **`xlwings.applescript` 在 wheel 内的确切相对路径/文件名**：researcher 证实该文件存在且 `runpython install` 即投递之，但 wheel 内路径未逐字核实 → 实施期 T1.3 以 `python -c "import xlwings, pathlib; print(pathlib.Path(xlwings.__file__).parent)"` 定位并断言存在。
2. **pkgbuild 用户域安装的 `--install-location` 语义**（相对路径映射到 `~/`）：R3 已裁定用户域安装，但用户域 pkg 的 install-location 语法细节（相对路径 vs 绝对路径映射）无本地实测 → M2 实现期以 productbuild/pkgbuild 干跑验证，风险 RK-08。
3. **V-01~V-05 五项真机行为**：官方文档存在灰色地带（详见 Validation contract §8.6，责任人=用户）。
4. **xstars-dev 旧脚本行号 ±5 漂移**：凡引用旧仓库行号，实施前打开原文件二次核对（本文开头已声明）。
5. **`.gitignore:10` 当前为 `*.xlsx` + `:40` `installer/XSTARS.xlam`**：已由 feature-planner 二次实读核对，与 cartographer 报告一致——此项缺口已闭环。

---

## 5. Gap analysis

| ID | 功能缺口 | 现状 | 影响 | 补齐任务 |
| --- | --- | --- | --- | --- |
| G1 | 无内置 Python 运行时组装链路（下载、SHA256 校验、解压、装依赖、装 xstars 非 editable） | 仓库无任何构建期运行时准备逻辑；全仓库 `*.xlam` 为空、无 LFS 制品治理（explore §2.2〔P〕） | 无法产出「零 Python」终端用户安装包 | T1.2, T1.3 |
| G2 | 无 pkg 构建链路（staging、pkgbuild、productbuild、distribution.xml 模板） | 当前仓库无 `installer/mac/`；旧链路在 xstars-dev 且含递归缺陷（explore §1.2〔P〕） | 构建不可重复，无法出 `.pkg` | T2.1, T2.2, T2.3 |
| G3 | 无 AppleDouble/`com.apple.provenance` 防护 | 旧 `build_pkg.sh:47-48`〔C〕有手法但未随仓库迁移；Sequoia 上无防护会导致 cpio 提取出 `._*` 伴生文件、目录损坏 | 安装期 payload 损坏 | T2.2, T2.4（断言） |
| G4 | 无安装期部署脚本（四件事投递 + Console User/属主修正 + 幂等） | 旧 `postinstall.sh` 服务旧进程模型（部署 `.app`+自研 `.scpt`），与 R16 四件事不符；且 root 写用户目录有属主陷阱（researcher §D） | 用户装完功能区不出现 / 文件属主错误 / Excel 沙盒读写失败 | T3.1, T3.2 |
| G5 | xlwings 解释器指向缺失（`INTERPRETER_MAC` 写入沙盒容器配置） | 当前仅开发者模式手工指 venv（`docs/macos-developer-setup.md:41-64`〔P〕）；终端用户无任何配置来源 | `RunPython` 找不到解释器，功能区按钮全部无效 | T3.2 |
| G6 | `XSTARS.xlam` / `XSTARS_mac.xlsm` 制品不在仓库 | `ribbon/README.md:6-8` 明示「不 ship 预构建 .xlsm/.xlam」（已核对）；Mac 无法脚本生成（无 VBProject） | 安装器无 UI 入口可投递、无开箱模板；CI 无法端到端闭环 | T0.1, T0.2, T0.3 |
| G7 | 无卸载与残留清理（`uninstall.sh`、`OPENn` 注册残留、收据 forget、备份 + `PRAGMA integrity_check`） | 旧版仅覆盖式 `rm -rf`（explore §5）；`docs/macos-developer-setup.md:191-193` 实证删文件后 Excel 每次启动弹「找不到加载项」 | 卸载后 Excel 每次启动报错，用户不可接受 | T4.1, T4.2 |
| G8 | 版本双源且不一致 | `pyproject.toml:7`=1.1.1 vs `xstars/__init__.py:3`=1.0.0（均已实读核对）；旧构建脚本读 `__init__.py`〔C〕 | 包名/收据/文档版本互相矛盾 | T1.1（构建读 pyproject）, T5.1（对齐 `__init__.py`） |
| G9 | `Pillow` 未显式声明 | `pyproject.toml` 无之，但 `xstars/main.py:1053`、`xstars/application/export.py:328,340` 直接使用（已实读核对） | 依赖图脆弱：matplotlib 换实现即断 | T5.1 |
| G10 | 7 处文档声明「macOS 仅开发者模式 / 无安装器」过时 | `README.md:185-188,244`、`README.zh-CN.md:184-187,243`、`docs/macos-developer-setup.md:3-5`、`docs/macos-manual-acceptance.md:3-5,60-67`、`docs/cross-platform-office-technology-strategy.md:18-24`、`ribbon/README.md:6-8,66`（cartographer §5 逐行摘录） | 文档与交付物矛盾，误导用户 | T5.2, T5.3, T5.4, T5.5 |
| G11 | CI 无 macOS 出包 job | `macos-support.yml` 仅测试三 job（已实读核对全文） | 构建不可重复、无产物留存 | T6.1 |
| G12 | 用户级免提权安装形态缺失 | 旧 `distribution.xml:9`〔C〕为 `enable_localSystem`（写 /Applications + root）；R3 要求 `enable_currentUserHome` + `~/Library/Application Support/XSTARS/` | 不满足 R3 即要求管理员密码，违背免提权承诺 | T2.1（distribution 模板）, T2.3（install-location）, T3.1（postinstall 适配用户上下文） |
| G13 | 安装器装配逻辑无测试面 | 仓库无任何 `installer/` 测试、无 shell 测试先例（cartographer §3.2） | 装配错误只能装机踩坑发现 | T1.4, T2.4, T3.3, T4.2（横切：支撑 G1–G12 的回归防线，续接 R20/pytest 约定） |
| G14 | 真机验收契约未覆盖安装器 | `docs/macos-manual-acceptance.md:3-5,60-67` 仅覆盖开发者模式；V-01~V-05 五项灰区无归属 | 安装器行为无验收闭环 | T6.2（横切：支撑 G4/G5/G6 的真机验证，R23） |
| G15 | 旧构建流水线递归缺陷（签名触发成环） | `build_pkg.sh`〔P/C〕+ `sign_and_notarize.sh:61`〔P〕无限递归；本轮不签名但须防移植复发 | 若原样移植且误设 `XSTARS_SIGN=1` → 构建机进程耗尽崩溃 | T2.3（单向流水线，无签名步骤、无回跑；R14） |
| G16 | 最低 macOS 版本约束未对齐依赖现实 | 旧 `distribution.xml:14`〔P〕min=11.0；scipy arm64 wheel 标签 `macosx_12_0_arm64`（researcher §A） | min=11.0 的包在 Big Sur/Monterey 以下装上但 scipy 不可用 | T2.1 |
| G17 | 架构声明与产物不符 | 旧 `distribution.xml:11`〔P〕声明 `arm64,x86_64` 但旧产物实为 arm64 thin〔P〕；R15 裁定 arm64-only | 声明与产物不符；Rosetta 机器误装 | T2.1 |

> 注：T0.3（assets README）支撑 G6 的制品可追溯性；T1.1 同时是 G2 的构建入口与 G8 的版本单源实现；基础验证类任务（T1.4/T2.4/T3.3/T4.2/T6.2）均已在表内注明支撑的缺口。

---

## 6. Milestone 表格

| Milestone | Status | Dependencies | Validation | Notes |
| --- | --- | --- | --- | --- |
| **M0 制品与前置就绪**（用户交付 xlam/xlsm 入库） | [x] **已完成（rev 6）**：T0.1 ✅（`20ba6c8`+`9b92364` 隐私清洗）、T0.2 ✅（`b9fac14`）、T0.3 ✅（README.md） | 无 | 两制品均在 `installer/mac/assets/`；静态断言全 PASS（2006 命名空间、无 customUI14/insertAfterMso、rel 包根挂载、三模块归一化一致、Interpreter 空、无 absPath）；pytest 文件随 T1.4 落地 | **M0 闭合，M1 解锁** |
| **M1 运行时组装**（下载 + 校验 + 装依赖 + 装 xstars） | [x] **已完成（rev 7，`aef6a32`）**：T1.1–T1.4 ✅ | M0 | staging 干跑：脱离仓库根 `import xstars, xlwings, ttkbootstrap, matplotlib, PIL` OK 且 `xstars.__file__` 在 staging site-packages；SHA256 篡改样本 → 非零退出；T1.4 单测 15/15 | 运行时 tarball 不入库（R1）；依赖装进主 site-packages；staging 目录已 gitignore（rev 7） |
| **M2 pkg 构建**（staging/pkgbuild/productbuild/distribution.xml） | [x] **已完成（rev 8，`f352a05`）** | M0, M1 | ✅ 真实出包 `XSTARS-1.1.1.pkg`；`xar`/`expandpkg` 结构验证；无 `._*`；unsigned 预期值；distribution 三参数断言；T2.4 单测全绿 | 单向流水线（无签名、无回跑，R14/G15）；AppleDouble 防护（R19/G3）；RK-08 实测解除 |
| **M3 安装期部署**（postinstall 四件事 + 权限/属主） | [x] **已完成（rev 9，`15536c1`）** | M1, M2 | ✅ postinstall 结构测试全绿（路径常数/四件事/幂等/退出码/conf 断言）；真实重出包含 Scripts/postinstall + xlwings.conf、零 AppleDouble；真机四件事由 V-02/V-03 覆盖（责任人=用户） | 用户域下以安装用户运行，属主逻辑保留双保险 |
| **M4 卸载与残留清理** | [x] **已完成（rev 10）**：uninstall.sh + 文档卸载章节 + payload 收入验证（T2.2 机制验证生效） | M3 | T4.2 静态/单元断言 31/31；`uninstall.sh --help`/默认 dry-run 可跑；文档含备份、`OPENn` 清理与 `PRAGMA integrity_check` 步骤（R7） | 首次一等公民卸载（G7）；真机破坏性验证归 8.4（责任人=用户） |
| **M5 文档同步 + 版本单源 + Pillow 声明** | [x] **已完成（rev 11，`c52f80c`）** | M0（可与 M2–M4 并行推进，合并前完成） | ✅ `xstars/__init__.py:3` == pyproject 版本（防漂移单测）；pyproject 含 `Pillow`；ruff + pytest + compileall 全绿；文档全文无「macOS 仅开发者模式/无安装器」孤立表述（grep 零命中） | G8 后半、G9、G10 |
| **M6 CI 出包 job + 真机验收回填** | [~] **进行中（rev 12）**：T6.1 ✅（`10cab2d`）；T6.2 ⬜ 待用户真机 V-01~V-05 | M1–M5 全部 | CI `macos-pkg-build` job 真实首跑（PR #6）产出 `.pkg` 并上传 artifact 成功；`git diff --exit-code origin/main...HEAD -- 'ribbon/*.bas'` 为空；V-01~V-05 由用户回填 Draft PR（R23） | 真机项责任人=用户（G14）；T6.2 完成后 M6 闭合 |

Milestone 总数 = **7**（≤10 上限，满足 ≤7 目标，无需合并说明）。

---

## 7. 分 Milestone 的 To-do checkbox 清单

### M0 制品与前置就绪（阻塞前置）

- [x] T0.1 ~~用户手工制作 `XSTARS.xlam` 并交付~~ **已完成（rev 4，2026-09-08）**：Agent 在 Windows 侧全自动制作（customUI 转换 + 真 Excel 16 COM + OOXML 注入，绕过 WPS CLSID 劫持经 ROT 绑定），入库至 `installer/mac/assets/XSTARS.xlam`（commit `20ba6c8`），验收断言全 PASS；含 D2 裁定的隐藏 `xlwings.conf` 兜底 sheet（`Interpreter` 留空）
  - 文件：新建 `installer/mac/assets/XSTARS.xlam`（二进制制品，常规 blob，R21）
  - 修改：无源码修改；由用户按 `ribbon/README.md:44-51` 在 Windows/RibbonX Editor 侧制作：以 2006-format customUI part（`customUI/customUI.xml`，命名空间 `http://schemas.microsoft.com/office/2006/01/customui`，去除全部 `insertAfterMso`）+ 未修改的 `ribbon_callbacks.bas` 副本，另存为 Excel Add-In（`.xlam`）后拷回
  - 验收：pytest 打开 xlam（zip 容器）断言：存在 `customUI/customUI.xml` 且命名空间为 2006 URI、无 `customUI14` part、无 `insertAfterMso` 属性；VBA 模块源文本与 `ribbon/ribbon_callbacks.bas` **行尾归一化（`\r\n`→`\n`）后逐字节一致**（D11 已裁定，rev 5；实测制品 163 行全 CRLF、仓库全 LF，差值恰为行数）；customUI relationship 挂载于包根 `_rels/.rels` 且 type 为 2006 `ui/extensibility`（与用户 Mac 实测渲染中的 `xlwings/addin/xlwings.xlam` 逐模式一致）
  - 依赖：无（用户动作，**阻塞 M1–M6 全部**）
- [x] T0.2 ~~用户手工制作~~ **已完成（rev 6，2026-09-08）**：无需 Windows 折返——用户在 Mac 侧从 DC2 验收工作簿另存净化，Agent 复验后入库（`b9fac14`）。复验：三模块归一化后与仓库 `.bas`/0.37.0 wheel 逐字节一致、`Dictionary.cls` 在位、`xlwings.conf!Interpreter` 空、`vbaProject.bin` 230,400B、全包无 `/Users/frank`/`venv` 泄漏（absPath 已清洗）
  - 文件：新建 `installer/mac/assets/XSTARS_mac.xlsm`（命名必须 `.xlsm`，`.gitignore:10` 会忽略 `.xlsx`，R21）
  - 修改：无源码修改；由用户按 `docs/macos-developer-setup.md:88-99` 制作：宏工作簿内嵌 `RibbonCallbacks`（导入未修改 `ribbon_callbacks.bas`）+ `xlwings` 模块（`xlwings.bas`，版本与内置运行时一致）+ `Dictionary` 类模块；保存为 `.xlsm`
  - 验收：pytest 断言 vbaProject.bin 存在；若工作簿含 `xlwings.conf` sheet 则其 `Interpreter` 项与 R16④ 路径一致或为空（兜底见待决 D2）；文件可被 `zipfile` 打开且宏项目非空
  - 依赖：T0.1（同一批用户交付物；内置运行时确定后核对 `xlwings.bas` 版本一致）
- [x] T0.3 **已完成（rev 6，2026-09-08）**：`installer/mac/assets/README.md` 落盘——制品用途矩阵、5 条硬不变量（2006 格式/归一化比对/xlwings 版本匹配/Interpreter 空/absPath 清洗）、Windows 与 Mac 双主机再制作步骤（含 WPS CLSID 劫持与 Mac VBE 拒 `.cls` 绕行）、再制作触发条件（RK-04）
  - 文件：新建 `installer/mac/assets/README.md`
  - 修改：纯新增文档
  - 验收：内容含 `ribbon/README.md:44-51` 与 `docs/macos-developer-setup.md:88-99` 的再制作引用，及「制品与 `.bas` 漂移时须重新制作」警示
  - 依赖：T0.1, T0.2（支撑 G6 制品可追溯）

### M1 运行时组装

- [x] T1.1 **已完成（rev 7，`aef6a32`）**：`installer/mac/build_pkg.py`（14.7KB，纯函数+注入式下载/命令执行器）；`--version` 输出 1.1.1 与 pyproject 一致；fail-closed 单测覆盖
  - 文件：新建 `installer/mac/build_pkg.py`
  - 修改：纯新增；`--prepare-runtime` / `--build-pkg` 子命令骨架；版本一律 `tomllib.load(pyproject.toml)["project"]["version"]`（R6，不移植旧 `_extract_version()`〔C〕正则方案）；所有装配步骤抽为纯函数（R20）
  - 验收：`build_pkg.py --version` 输出 1.1.1 且与 `pyproject.toml:7` 一致；单测覆盖版本读取函数（含 pyproject 缺失/损坏时 fail-closed 非零退出）
  - 依赖：无（支撑 G2 入口、G8）
- [x] T1.2 **已完成（rev 7）**：lock 字段齐备（url/sha256/size/python_version/arch/variant/tag，D1 钉值 `20260901`/3.12.14）；篡改样本 → 非零退出；下载器可注入 fake urlopen
  - 文件：新建 `installer/mac/runtime.lock.json`；`installer/mac/build_pkg.py`（新增下载/校验函数）
  - 修改：lock 文件含 `python_build_standalone` 下载 URL、SHA256（**具体钉值实施时从 astral-sh/python-build-standalone Releases 页取**，待决 D1）、目标 Python 版本（3.12，R2）、架构（aarch64-apple-darwin）、变体（install_only_stripped，R1）；下载后 `hashlib.sha256` 逐块校验，不匹配 → 打印期望/实际并 `sys.exit(1)`（fail-closed，R1）
  - 验收：单测用篡改样本校验失败路径（exit 非 0）；网络下载函数可被注入 fake urlopen 测试；lock 文件被 pytest 断言字段齐备
  - 依赖：T1.1（支撑 G1）
- [x] T1.3 **已完成（rev 7）**：真实干跑通过——staging `python/bin/python3 -c "import xstars, xlwings, ttkbootstrap, matplotlib, PIL"` OK（脱离仓库根执行，`xstars.__file__` 在 staging site-packages）；`pip list` 无 dev 项；`xlwings.applescript` 已汇集至 staging/bin
  - 文件：`installer/mac/build_pkg.py`（新增 staging 组装函数）
  - 修改：解压 tarball 至 `staging/python/`（可重定位，§4.3）；用该解释器执行 `pip install --no-cache-dir <repo>`（xstars 本体，**非 editable**，R16①）+ 全部运行时依赖（来自 pyproject dependencies，自动含 ttkbootstrap；**不含 dev extras**；`xlwings` wheel 的 mac 侧原生依赖 psutil/appscript 随 pip 解析）；定位 site-packages 内 `xlwings.applescript` 文件并断言存在（§4.5 缺口 1 的实施期核实）；产出 `staging/python/`、`staging/bin/`（xlam/applescript/xlsm 汇集）
  - 验收：staging 内 `python/bin/python3 -c "import xstars, xlwings, ttkbootstrap, matplotlib, PIL"` 成功且 `xstars.__file__` 位于 staging site-packages（非源码目录）；`pip list` 无 pytest/dev 项；`xlwings.applescript` 已复制到 `staging/bin/`
  - 依赖：T0.1, T0.2, T1.2（支撑 G1、G6 汇集）
- [x] T1.4 **已完成（rev 7）**：15 项单测全绿（版本 fail-closed/SHA 篡改/staging 布局/xlam 全断言含 rel 包根挂载/xlsm 三模块+D11 归一化比对/占位符规则 `/Users/(?!<User>)`）；oletools 入 `[dev]`
  - 文件：新建 `tests/test_macos_installer.py`
  - 修改：纯新增；沿用仓库 pytest 约定（cartographer §3.3：能 mock 则 monkeypatch，不打 `skipif`）
  - 验收：覆盖：版本读取、SHA256 校验失败 fail-closed、staging 布局断言、xlam 2006 命名空间断言（T0.1 验收落点）、xlsm 宏项目断言（T0.2 验收落点）；本地 `pytest tests/test_macos_installer.py` 全绿
  - 依赖：T1.1, T1.2, T1.3, T0.1, T0.2（横切 G13）

### M2 pkg 构建

- [x] T2.1 **已完成（rev 8，`f352a05`）**：`distribution.xml` 四项改造落地（arm64/min 12.0/currentUserHome/**移除 rootVolumeOnly**——与用户域 home 安装互斥）；构建期 Python 渲染替代 sed
  - 文件：新建 `installer/mac/distribution.xml`
  - 修改：以旧版为基（引用时实施前二次核对旧文件行号）：`hostArchitectures="arm64"`（R15/G17）；`<domains enable_currentUserHome="true"/>`（R3/G12，替换 `enable_localSystem`）；`<os-version min="12.0"/>`（R9/G16）；保留 `__VERSION__` 占位符与 `customize="never" require-scripts="true"` 结构
  - 验收：pytest 断言生成后的 XML 三项参数正确、`__VERSION__` 替换无残留
  - 依赖：T1.1（版本注入）
- [x] T2.2 **已完成（rev 8）**：payload 相对布局（`Library/Application Support/XSTARS/`）+ tarball 直通防护；uninstall.sh 容忍缺失（M4）；扫描无 `._*`/`.DS_Store`/制品 absPath
  - 文件：`installer/mac/build_pkg.py`（新增 payload 装配函数）
  - 修改：payload 布局 `~/Library/Application Support/XSTARS/` → `python/`、`bin/XSTARS.xlam`、`bin/xlwings.applescript`、`bin/XSTARS_mac.xlsm`、`uninstall.sh`、`Templates/XSTARS_mac.xlsm`（模板投放位置见待决 D7）；staging 阶段沿用 `COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata` 手法打包需经 tar 的内容（R19/G3，源自旧 `build_pkg.sh:47-48`〔C〕）；装配完成后扫描 payload 断言无 `._*` / `.DS_Store` 文件，且两二进制制品复扫无 `x15ac:absPath` 泄漏（rev 6 新增：M0 制品曾含 `/Users/frank`/`C:\Users\<user>` 构建路径元数据，入库前已外科手术清洗）
  - 验收：单测断言 payload 树结构与禁入文件扫描；干跑产出目录含全部四类内容物
  - 依赖：T1.3, T2.1
- [x] T2.3 **已完成（rev 8）**：pkgbuild/productbuild 单向流水线；真实出包 189,756,535 B unsigned；`XSTARS_SIGN` 显式忽略；RK-08 实测落位正确
  - 文件：`installer/mac/build_pkg.py`（新增打包函数与 CLI 主流程）
  - 修改：`pkgbuild --root <staging> --identifier com.frank-sysu.xstars --version <pyproject 版本> --install-location <用户域映射，见 §4.5 缺口 2，实施期干跑确定> --scripts <scripts>`；`productbuild --distribution <渲染后 XML> --package-path <work> installer/output/XSTARS-<version>.pkg`（`installer/output/` 已被 `.gitignore:39` 忽略，不污染仓库）；**无任何签名步骤、无任何脚本回跑**（R14/G15：不移植 `sign_and_notarize.sh`；`XSTARS_SIGN` 环境变量即使误设也被显式忽略并在日志注明）
  - 验收：`--build-pkg` 产出 `installer/output/XSTARS-1.1.1.pkg`；`xar -t -f` 列出预期条目；产物不含签名段（`pkgutil --check-signature` 报 unsigned，预期值）
  - 依赖：T2.2（支撑 G2、G12、G15）
- [x] T2.4 **已完成（rev 8）**：+8 项 M2 断言（23/23）；macOS 本机 pkg 结构断言落地
  - 文件：`tests/test_macos_installer.py`
  - 修改：新增：distribution.xml 三参数断言（T2.1 落点）；payload 树/AppleDouble 扫描断言（T2.2 落点）；打包命令行参数构造断言（纯函数，命令可注入，Linux runner 可跑）；（macOS 本机）pkg 结构断言
  - 验收：pytest 全绿；CI 非 macOS runner 亦可跑装配断言（命令注入替身，R20）
  - 依赖：T2.1, T2.2, T2.3（横切 G13）

### M3 安装期部署

- [x] T3.1 **已完成（rev 9，`15536c1`）**：postinstall.sh 201 行幂等——四件事全部落地（解包 fail-closed/Startup 预创建/Application Scripts/Containers conf）、Console User+scutil 兜底+属主双保险、无 rm -rf、不碰 ~/.xstars（否定断言）
  - 文件：新建 `installer/mac/postinstall.sh`
  - 修改：结构参考旧版（`set -e`、Console User 探测 `stat -f '%Su' /dev/console` + `dscl` Home 查询、非致命降级输出、末尾 `exit 0`——行号引用见 §4.1，实施前二次核对）。四件事：① 运行时由 Installer 按 install-location 自动落位（postinstall 仅断言存在，缺失则致命 exit 1）；② `cp bin/XSTARS.xlam` → `~/Library/Group Containers/UBF8T346G9.Office/User Content.localized/Startup.localized/Excel/`（目录不存在则 `mkdir -p` 预创建，V-02 真机项）；③ `cp bin/xlwings.applescript` → `~/Library/Application Scripts/com.microsoft.Excel/`；④ 写 `"INTERPRETER_MAC","$HOME/Library/Application Support/XSTARS/python/bin/python3"` → `~/Library/Containers/com.microsoft.Excel/Data/xlwings.conf`（`$HOME` 变量由 xlwings 展开，R16④）；属主：用户域安装下以用户运行即天然正确，但保留 Console User/chown 逻辑作双保险（root 上下文安装时不属 root:wheel，§4.3 TCC 行）；全部步骤幂等（重复安装覆盖写）
  - 验收：T3.3 静态断言通过；真机由 V-02/V-03 覆盖（用户）
  - 依赖：T2.3（payload 结构定型）
- [x] T3.2 **已完成（rev 9）**：`render_xlwings_conf()` 纯函数（保留未知键仅更新 INTERPRETER_MAC，RK-02）；scripts 目录已接线 pkgbuild
  - 文件：`installer/mac/postinstall.sh`（bash 落笔）；`installer/mac/build_pkg.py`（`render_xlwings_conf()` 纯函数）
  - 修改：conf 格式为 xlwings CSV 风格键值对：`"INTERPRETER_MAC","$HOME/Library/Application Support/XSTARS/python/bin/python3"`（大写键名、`$HOME` 变量，§4.3 xlwings 行）；已有 conf 时策略：保留未知键、仅更新 `INTERPRETER_MAC`（不整文件覆盖，防 Excel 首启重建冲突，RK-02）
  - 验收：pytest 断言 `render_xlwings_conf()` 输出格式与既有 conf 合并逻辑（未知键保留）
  - 依赖：T3.1（支撑 G5）
- [x] T3.3 **已完成（rev 9）**：+5 项 M3 断言（路径常数/幂等/退出码策略/属主逻辑/conf 合并；共 28 项）
  - 文件：`tests/test_macos_installer.py`
  - 修改：新增：postinstall 文本静态断言——四个目标路径常数齐备、`set -e` 存在、幂等（无 `rm -rf` 用户数据目录语句）、退出码策略（payload 缺失 exit 1 / 用户级拷贝降级不致命）、`chown`/Console User 逻辑存在；`render_xlwings_conf()` 单测
  - 验收：pytest 全绿
  - 依赖：T3.1, T3.2（横切 G13）

### M4 卸载与残留清理

- [x] T4.1 **已完成（rev 10）**：uninstall.sh 250 行——默认 dry-run/`--apply` 真删；六步清理（Startup xlam → Application Scripts 双脚本 → conf 仅删 INTERPRETER_MAC 行 → RegistrationDB `OPENn` 备份+integrity_check 失败即回滚 → 安装目录删（不碰 `~/.xstars`）→ `pkgutil --forget`）；Excel 运行探测拒执行
  - 文件：新建 `installer/mac/uninstall.sh`（同时由 T2.2 收入 payload → `~/Library/Application Support/XSTARS/uninstall.sh`）
  - 修改：清理清单：① Excel 启动项 `XSTARS.xlam`（Group Containers Startup 目录，删前确认存在）；② `~/Library/Application Scripts/com.microsoft.Excel/xlwings.applescript`；③ `~/Library/Containers/com.microsoft.Excel/Data/xlwings.conf` 中的 `INTERPRETER_MAC` 行（保留其余键）；④ **Excel 注册残留**：`~/Library/Group Containers/UBF8T346G9.Office/MicrosoftRegistrationDB/*.reg` 中 `OPENn` 记录清理——**先备份原 .reg 文件到带时间戳目录，再用 sqlite3 删除 Excel options 节点下指向 XSTARS.xlam 的 `OPENn` 行，删后 `PRAGMA integrity_check` 必须返回 `ok` 否则恢复备份并报错退出**（R7，依据 `docs/macos-developer-setup.md:191-193` 已核对）；⑤ `~/Library/Application Support/XSTARS/` 整目录删除（**`~/.xstars/` 用户数据默认保留**，见待决 D4）；⑥ `pkgutil --forget com.frank-sysu.xstars`（收据标识，注意待决 D9 的 identifier 沿用问题）；仅当 Excel 未运行时执行（探测到运行则提示退出）
  - 验收：T4.2 断言通过；`--help` 与 dry-run 模式可用；真机由用户验证一次完整卸载后 Excel 启动无弹窗
  - 依赖：T3.1（部署物清单对齐）
- [x] T4.2 **已完成（rev 10）**：静态断言（备份/integrity_check/forget/OPENn/`~/.xstars` 否定断言/dry-run 与 `--apply` 双模式/Excel 探测）全绿；payload 含 uninstall.sh（T2.2“存在才收入”验证生效）；`docs/macos-installer.md` 卸载章节先行
  - 文件：`tests/test_macos_installer.py`；`docs/macos-installer.md`（卸载章节，与 T5.2 同文件）
  - 修改：静态断言 uninstall 脚本含：备份命令、`PRAGMA integrity_check`、`pkgutil --forget`、`OPENn` 清理、`~/.xstars` 不被删除的否定断言；文档含逐步卸载指引与失败恢复（从备份还原 .reg）
  - 验收：pytest 全绿；文档评审通过（含 R7 全部要素）
  - 依赖：T4.1（横切 G13/G7）

### M5 文档同步 + 版本单源 + Pillow 声明

- [x] T5.1 **已完成（rev 11，`c52f80c`）**：`__init__.py` 1.1.1（仅一行）+ 防漂移单测；`Pillow>=10.0` 显式声明
  - 文件：修改 `xstars/__init__.py`（`:3` `"1.0.0"` → `"1.1.1"`）；修改 `pyproject.toml`（dependencies 增 `"Pillow>=10.0"`，具体下限实施时以当前 matplotlib 解析版本为准）
  - 修改：仅此两处顺手修（R6/R10，决策 10「仅此一项纳入」）；**不动 `requirements.txt`**（Non-goal 8）
  - 验收：新增单测断言 `xstars.__version__ == tomllib 读 pyproject 版本`（防再漂移）；pytest/ruff/compileall 全绿
  - 依赖：无硬依赖（支撑 G8 后半、G9）
- [x] T5.2 **已完成（rev 11）**：文档 147 行——系统要求/双 CLI target（含 M2 实测 CurrentUserHomeDirectory 语义）/三途放行（无右键打开话术）/已知限制含 quickstart.xlsm 上游元数据残余；卸载章节衔接 --apply
  - 文件：新建 `docs/macos-installer.md`
  - 修改：内容必须含：系统要求（Apple Silicon、macOS 12+、Excel for Mac 2016+，R9）；安装步骤（双击 / `sudo installer -pkg ... -target /`）；**未签名放行三途**（Sequoia 已移除右键打开：「系统设置 → 隐私与安全性 → 仍要打开」+ 密码 / `xattr -d com.apple.quarantine <pkg>` / `sudo installer`，§4.3 Sequoia 行）；卸载指引（引用 `~/Library/Application Support/XSTARS/uninstall.sh`，含备份与 integrity_check 说明）；已知限制（arm64-only、不支持 WPS for Mac、`~/.xstars` 数据保留策略）
  - 验收：三途放行指引齐全且不含「右键打开」过时话术；与 R7/R15/R9 对齐
  - 依赖：T4.1（卸载脚本定型）
- [x] T5.3 **已完成（rev 11）**：双语 README 安装包模式为推荐路径、开发者模式降级 fallback（`184` “source-debugging fallback”）；系统要求行分叉；过时表述 grep 零命中
  - 文件：修改 `README.md`（`:185-188, :244`）；修改 `README.zh-CN.md`（`:184-187, :243`）（cartographer §5.1/5.2 摘录的原文段）
  - 修改：新增「macOS Installer」章节（双击 `.pkg` 即装、免 Python、指向 `docs/macos-installer.md`）；原「Developer Mode」段降级保留为源码调试备选；系统要求行改为「安装包模式（Apple Silicon arm64, macOS 12+, Excel 2016+, 无需 Python）/ 开发者模式（Python ≥ 3.10）」双语对齐
  - 验收：两 README 表述同步；grep 无「developer mode only」孤立残留
  - 依赖：T5.2（链接目标存在）
- [x] T5.4 **已完成（rev 11）**：developer-setup 前言指引（正文全保留）；manual-acceptance 增安装器验收组 + I-01~I-04 标注适用域；strategy 路线表「独立 `.pkg` 安装器交付中」
  - 文件：修改三文件（锚点：developer-setup `:3-5` 前言增补「终端用户请用 .pkg 安装器，本文档仅供源码开发」；manual-acceptance `:3-5, :60-67` 增补「安装器验收组」并把 I-01~I-04 标注为 Developer Mode 适用；strategy `:18-24` 路线表「Excel macOS」状态更新为「独立 .pkg 安装器交付中」）
  - 修改：不删除开发者模式内容（保留为 fallback，cartographer §5.3/5.4 处置标记）
  - 验收：三文件更新后与交付物一致；manual-acceptance 新增安装器检查组条目
  - 依赖：T5.2
- [x] T5.5 **已完成（rev 11）**：ship 制品澄清 + 手工流程保留为开发者备忘；“does not ship a prebuilt” grep 零命中
  - 文件：修改 `ribbon/README.md`（`:6-8`「does **not** ship a prebuilt `.xlsm` or `.xlam`」与 `:66`「No standalone macOS `.app` is provided」两段，均已核对原文）
  - 修改：澄清：仓库现经 `installer/mac/assets/` ship 预构建 `XSTARS.xlam` 与 `XSTARS_mac.xlsm`（供 `.pkg` 使用）；仍无 `.app`，但提供 `.pkg` 完整安装包；手工 RibbonX Editor 流程保留为开发者备忘
  - 验收：表述与 R4/R21 一致；grep 无「does not ship a prebuilt」残留
  - 依赖：T0.1, T0.2（支撑 G10）

### M6 CI 出包 job + 真机验收回填

- [x] T6.1 **已完成（rev 12，`10cab2d`）**：`macos-pkg-build` job（macos-latest arm64 + 3.12 宿主 + 两步构建 + 测试 + compileall + artifact 7d；无 secrets）；本机全步骤预演退出码 0
  - 文件：修改 `.github/workflows/macos-support.yml`
  - 修改：新增 job：`runs-on: macos-latest`；checkout → setup-python 3.12（构建脚本宿主解释器）→ `pip install -e ".[dev]"` → `python installer/mac/build_pkg.py --build-pkg`（含运行时下载 + SHA256 校验 + 依赖安装）→ `python -m pytest tests/test_macos_installer.py` → `python -m compileall -q xstars tests installer/mac`（compileall 范围扩入 installer/mac）→ `actions/upload-artifact@v4` 上传 `installer/output/XSTARS-*.pkg`（retention-days 见待决 D6）；既有 vba-immutability job 不动（R12 门禁续跑）；既有测试 job 不动
  - 验收：PR 上该 job 绿且 artifact 含 `.pkg`；`git diff --exit-code origin/main...HEAD -- 'ribbon/*.bas'` 仍为空
  - 依赖：T1.1–T4.2 全部（支撑 G11，横切 G14 准备）
- [ ] T6.2 用户真机执行 V-01~V-05 并回填 Draft PR
  - 文件：Draft PR 描述（真机验收清单区）；必要时回填 `docs/macos-installer.md` 的实测勘误
  - 修改：用户在 Mac mini 执行 V-01~V-05（清单见 §8.6），逐项记录结果至 Draft PR；任何一项 Fail → 回到对应 milestone 修复后再验
  - 验收：Draft PR 内 V-01~V-05 五项均有 Pass/Fail 记录与证据；全部 Pass 方可转 Ready for review
  - 依赖：T6.1（拿到 CI 产物 pkg；用户也可本机 `--build-pkg` 产出）（支撑 G14，R23）

---

## 8. Validation contract

### 8.1 静态检查与全量回归（每次 PR push，CI + 本地）

| 检查项 | 命令/方式 | 预期结果 | 通过标准 |
| --- | --- | --- | --- |
| 全量 pytest 回归 | `python -m pytest` | 基线量级：main 上 353 passed / 13 skipped（**以实际运行为准**，本 Plan 未实测基线） | 无新增 failed；新增测试全绿 |
| ruff | `ruff check .`（staging 已 gitignore 排除） | 新增文件 0 error（`installer/mac/build_pkg.py`、`tests/test_macos_installer.py`）✅；全仓存量 117 error（`xstars/main.py` 34 等）为 main 从未 ruff-clean 的先存债，相关文件在「明确不修改」清单，不在本 Plan 范围 | 新文件 0 error；CI 不跑 ruff（既有事实） |
| compileall | `python -m compileall -q xstars tests installer/mac` | 静默成功 | 退出码 0 |
| 安装器单测 | `python -m pytest tests/test_macos_installer.py -v` | 覆盖：版本单源、SHA256 fail-closed、staging/payload 布局、AppleDouble 扫描、distribution.xml 三参数、postinstall/uninstall 静态断言、xlam 2006 命名空间、xlam customUI rel 包根挂载（2006 type，对照 `xlwings/addin/xlwings.xlam` 已验证模式）、VBA 源行尾归一化比对（D11，VBA 提取用 oletools，已验证可行；T1.4 决定是否入 `[dev]`）、xlsm 宏项目、`render_xlwings_conf()` | 全绿 |
| **VBA 零修改门禁** | `git diff --exit-code origin/main...HEAD -- 'ribbon/*.bas'` | **必须为空**（`macos-support.yml:73` 既有 job） | diff 为空，job 绿 |
| 版本单源 | 新增单测断言 `xstars.__version__` == pyproject 版本 | 1.1.1 == 1.1.1 | 断言通过 |

### 8.2 pkg 产物结构断言（macOS 本机 / CI macos-latest）

| 检查项 | 命令/方式 | 预期结果 | 通过标准 |
| --- | --- | --- | --- |
| 归档结构 | `xar -t -f installer/output/XSTARS-<ver>.pkg` | 列出 Payload、Scripts 等预期条目 | 条目齐备 |
| Payload 内容 | `pkgutil --expandpkg <pkg> <dir> && find <dir>/Payload -type f` | 含 `python/bin/python3`、site-packages（xstars/xlwings/ttkbootstrap 等）、`bin/XSTARS.xlam`、`bin/xlwings.applescript`、`bin/XSTARS_mac.xlsm`、`uninstall.sh` | 全部存在 |
| **无 AppleDouble（外层）** | 展开后 `find <dir> -name '._*' -o -name '.DS_Store'` | **零命中**（R19/G3） | 空输出 |
| **无 AppleDouble（内层 Payload，rev 10 增）** | `gzip -dc Payload \| cpio -it \| grep -c '\._'` | **0**（rev 10：M3 曾仅查外层，内层实测 4 条已修复） | 0 |
| 未签名状态 | `pkgutil --check-signature <pkg>` | 报 unsigned / no signature（**预期值**，R14） | 与预期一致（非错误） |
| 干跑安装 | `sudo installer -pkg <pkg> -target ~`（或用户域等价目标，见 §4.5 缺口 2）——**仅限用户自愿的测试机** | postinstall 日志四件事齐备、`exit 0` | 日志含 4 个 deployed 项 |

### 8.3 构建脚本行为断言

| 检查项 | 方式 | 通过标准 |
| --- | --- | --- |
| SHA256 fail-closed | 单测注入篡改 tarball | 非零退出 + 明确错误信息 |
| 运行时可执行 | staging 内 `python/bin/python3 -c "import xstars, xlwings, ttkbootstrap, PIL; print('OK')"` | 输出 OK；`xstars` 来自 staging site-packages |
| 单向流水线 | 全文审计 build_pkg.py 无对自身/其他脚本的递归调用；`XSTARS_SIGN` 被忽略 | 无回跑路径（G15 防复发） |

### 8.4 卸载验证（真机，用户）

| 检查项 | 方式 | 通过标准 |
| --- | --- | --- |
| 完整卸载 | 真机跑 `~/Library/Application Support/XSTARS/uninstall.sh`，随后启动 Excel 3 次 | 无「找不到加载项」弹窗；Startup 目录无 XSTARS.xlam；.reg 备份存在且 `PRAGMA integrity_check` ok |
| 数据兼容 | 卸载后检查 `~/.xstars/` | settings.json 与 artifacts 仍在（默认保留，待决 D4） |

### 8.5 人工验证（无法自动化项）

- M0 制品人工环节（xlam/xlsm 制作）：**物理不可自动**（Excel for Mac 无 VBProject，Non-goal 6）；责任人 = 用户（T0.1/T0.2 制作）+ Agent（入库与 pytest 断言）。
- Excel 宿主内功能区渲染与 RunPython 触发：CI 不启动 Excel（`docs/macos-manual-acceptance.md:3-5` 约定）；责任人 = 用户。

### 8.6 真机验证 V-01~V-05（外部调研灰区项，责任人 = 用户，R23）

| ID | 验证项 | 验证方法 | 通过标准 |
| --- | --- | --- | --- |
| V-01 | python-build-standalone 内置 Tkinter 对 ttkbootstrap/matplotlib 开箱可用性（风险：`Can't find a usable init.tcl`） | 干净 arm64 终端：`<runtime>/bin/python3 -c "import ttkbootstrap as tb; root=tb.Window(); root.destroy(); print('TK_OK')"` | 输出 TK_OK；若失败则按 RK-01 缓解（TCL_LIBRARY/TK_LIBRARY sitecustomize）后复测通过 |
| V-02 | 首次未运行 Excel 时目录预创建有效性 | 全新用户状态下直接安装（postinstall `mkdir -p` 预创建三目录），首次启动 Excel | XSTARS 功能区出现；预创建目录/文件未被 Excel 覆盖删除 |
| V-03 | `INTERPRETER_MAC` 沙盒配置读取有效性 | 点 Ribbon 任一按钮触发 RunPython，观察 python3 进程路径 | 进程路径为 `~/Library/Application Support/XSTARS/python/bin/python3`；无权限弹窗 |
| V-04 | quarantine 弹窗与放行路径（macOS 15.3+） | 将 pkg 经网络途径传输后双击，走「系统设置 → 隐私与安全性 → 仍要打开」 | 放行成功进入安装向导；记录每步截图回填文档 |
| V-05 | RunPython 反复触发稳定性 | 连续点 Ribbon 按钮 5 次执行测试任务，Activity Monitor 观察 python3 生命周期 | 无僵尸进程、无 Office 卡死 |

> 8.4–8.6 当前**无法由 Agent 执行**（无真实 Excel/macOS 宿主，R23/Non-goal 12）；责任人 = 用户，结果回填 Draft PR（T6.2）。

---

## 9. 文件级修改范围 + 风险 / 回滚 / 待决事项 + Git 策略

### 9.1 文件级修改范围

**新建（13）**：

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `installer/mac/build_pkg.py` | 源码（纯函数 + CLI） | 构建入口：版本读取/运行时下载校验/staging/payload/pkgbuild/productbuild（T1.1–T2.3） |
| `installer/mac/runtime.lock.json` | 配置（新） | python-build-standalone URL + SHA256 钉值 + 版本/架构/变体（T1.2） |
| `installer/mac/distribution.xml` | 模板 | arm64-only / min 12.0 / `enable_currentUserHome` / `__VERSION__`（T2.1） |
| `installer/mac/postinstall.sh` | 脚本（薄 bash） | 四件事投递 + 属主修正（T3.1/T3.2） |
| `installer/mac/uninstall.sh` | 脚本（薄 bash） | 卸载 + OPENn 清理 + 备份 + integrity_check（T4.1） |
| `installer/mac/assets/XSTARS.xlam` | **二进制制品**（用户制作交付） | 2006-format ribbon 加载项（T0.1） |
| `installer/mac/assets/XSTARS_mac.xlsm` | **二进制制品**（用户制作交付） | 预置模板（T0.2） |
| `installer/mac/assets/README.md` | 文档 | 制品来源与再制作说明（T0.3） |
| `tests/test_macos_installer.py` | 测试 | 全部安装器断言（T1.4/T2.4/T3.3/T4.2） |
| `docs/macos-installer.md` | 文档 | 安装/放行/卸载用户文档（T5.2，含 T4.2 卸载章节） |
| `plans/` 下本 Plan 及（后续）验收记录 | 规划文档 | 已落盘本文件 |

**修改（8）**：

| 文件 | 修改点 |
| --- | --- |
| `pyproject.toml` | dependencies 增 `Pillow`（R10）；**仅此一处** |
| `xstars/__init__.py` | `:3` `"1.0.0"` → `"1.1.1"`（R6） |
| `.github/workflows/macos-support.yml` | 新增 `macos-pkg-build` job + compileall 范围扩入 `installer/mac`（T6.1）；既有三 job 不动 |
| `README.md` | `:185-188, :244` macOS 章节改写（T5.3） |
| `README.zh-CN.md` | `:184-187, :243` 同步改写（T5.3） |
| `docs/macos-developer-setup.md` | `:3-5` 前言增补安装器指引；正文开发者流程保留（T5.4） |
| `docs/macos-manual-acceptance.md` | 增补安装器验收组；I-01~I-04 标注 Developer Mode 适用域（T5.4） |
| `docs/cross-platform-office-technology-strategy.md` | `:18-24` 路线表状态更新（T5.4） |
| `ribbon/README.md` | `:6-8, :66` ship 制品与 `.pkg` 表述更新（T5.5） |

（修改行号来自 cartographer §5 摘录，实施前以当前 HEAD 实读为准。）

**明确不修改（10）**：

| 文件 | 理由 |
| --- | --- |
| `ribbon/ribbon_callbacks.bas` | R12 硬门禁（CI `:73` 零差异） |
| `ribbon/ribbon_callbacks_installed.bas` | 同上 |
| `ribbon/ribbon_callbacks_standalone.bas` | 同上 |
| `xstars/main.py` | 业务代码零改动——自带运行时是当前架构原生形态（§4.1）；darwin 分支已在位 |
| `installer/wps/*` | 产品线隔离（installer/<product>/ 约定）；缓存目录不清理（Non-goal 10） |
| `xstars/cli.py` | 不恢复未保存工作簿回退（Non-goal 9） |
| `requirements.txt` | 陈旧子集不修（Non-goal 8；CI 未使用） |
| `.gitignore` | 已核实制品路径不会被忽略（R21），无需改动 |
| `.gitattributes` | **rev 3 修改**：追加 `*.bas text eol=lf`（R24）；OOXML/GIF 二进制经 `git ls-files --eol` 实测为 `-text`，无需 `binary` 声明 |
| `ribbon/customUI14.xml` | 打包格式转换在制品制作侧完成（T0.1），源文件不动 |
| `docs/wps-installation.md` | 仅作约定参照（R3），不修改 |

**删除**：无。

### 9.2 风险

| ID | 风险 | 等级 | 触发条件 | 影响 | 缓解 |
| --- | --- | --- | --- | --- | --- |
| RK-01 | Tk 在 standalone 运行时找不到 `init.tcl`（python-build-standalone 已知 Quirk） | 中 | 特定安装路径/嵌套环境组合 | ttkbootstrap/matplotlib TkAgg 弹窗崩溃，功能区按钮报错 | build 期生成 sitecustomize 预设 `TCL_LIBRARY`/`TK_LIBRARY` 相对路径（§4.3 quirks 行）；V-01 真机首验 |
| RK-02 | 用户级安装后 Excel 沙盒读 `Containers` 配置被覆盖/重建 | 中 | Excel 首启晚于 postinstall（V-02 灰区） | `INTERPRETER_MAC` 丢失 → RunPython 失效 | T3.2 仅更新键不整文件覆盖；可选兜底 = xlam 内置 xlwings.conf sheet（待决 D2）；V-02/V-03 真机验证 |
| RK-03 | 未签名 pkg 跨机分发被 Gatekeeper 拦（Sequoia 无右键打开） | 高（跨机分发时） | 经网络途径传输（带 quarantine） | 终端用户双击被拦 | 文档三途放行指引（T5.2）；`sudo installer` 一键路径；V-04 真机验证；本机构建/scp/USB 不受影响 |
| RK-04 | 制品（xlam/xlsm）与 `ribbon/*.bas` 版本漂移 | 中 | `.bas` 后续演进而制品未重做 | 功能区回调与代码不匹配、静默失效 | T0.1 验收含「模块源文本与 .bas 逐字节一致」pytest 断言（制品入库后每次 CI 自动比对）；T0.3 README 警示重做义务；rev 3：R24 已钉 `*.bas eol=lf` 消除 Windows 检出 CRLF 源；rev 5：D11 已裁定——断言采用行尾归一化（VBE 在 vbaProject.bin 内固定 CRLF 存储，实测归一化后逐字节一致） |
| RK-05 | 运行时体积导致 CI artifact 过大 | 中 | runtime + scipy/pandas/matplotlib 等 site-packages 解压后数百 MB | CI 慢、artifact 超限 | install_only_stripped 变体（R1）；CI 仅上传最终 .pkg（压缩态）；retention-days 待决 D6；必要时 `pip install --no-cache-dir` + 剔除 `__pycache__`/tests |
| RK-06 | 多用户机器每人一份 | 低 | 同机多 GUI 用户 | 其他用户无功能区（仅安装者可见） | 与 R3 用户级免提权裁定一致的既知取舍；文档写明「每个需要的用户各装一次」；不试图做系统级（违背裁定） |
| RK-07 | 覆盖安装/升级时 `~/.xstars` 用户数据 | 中 | 重装或升级 pkg | 若误删则用户设置/工件丢失 | uninstall.sh/postinstall 均不触碰 `~/.xstars`（T3.3/T4.2 否定断言）；保留策略见待决 D4 |
| RK-08 | pkgbuild 用户域 `--install-location` 语义实现细节不符预期 | 中 → **已消除（rev 8 实测）** | productbuild 对用户域相对路径映射与预期不符 | 安装位置错误或要求提权 | **实测解除**：`--install-location /` + payload 相对布局 + `-target CurrentUserHomeDirectory` → 实际落位 `~/Library/Application Support/XSTARS/`（测试安装 + receipt 已清理） |
| RK-09 | SHA256 钉值对应的 3.12.x 构建过时（安全更新滞后） | 低 | 长期不更新 lock | 内置 Python 含已知 CVE | lock 文件集中管理、升级仅需改一处 + 重跑校验；文档注明运行时版本；不在本轮范围（签名/更新通道属后续） |
| RK-10 | postinstall 在无 GUI 登录场景（SSH/远程安装）探测不到 Console User | 低 | 远程静默安装 | 用户目录投递被跳过 | 沿用旧版「跳过 + 指引输出、exit 0」非致命策略（§4.1 postinstall 段）；文档写明需登录图形会话安装 |
| RK-11 | `com.frank-sysu.xstars` 收据与旧 xstars-dev 安装残留冲突 | 低 | 机器曾装旧 pkg | `pkgutil --forget`/升级行为混乱 | 待决 D9（identifier 是否换新）；uninstall 文档含旧收据检查步骤 |

### 9.3 回滚

- **代码回滚**：单分支单 PR → `git revert` 合并提交即可整体回退；无数据库/格式迁移。
- **安装回滚（终端用户）**：运行 `~/Library/Application Support/XSTARS/uninstall.sh`（或仓库副本）→ 清理四类部署物 + OPENn 注册残留 → `pkgutil --forget com.frank-sysu.xstars` 移除收据 → Excel 重启无弹窗。
- **数据兼容性**：`~/.xstars/`（`settings.json`、`artifacts/`）在安装、覆盖安装、卸载全程**默认不触碰**（RK-07；`xstars/config.py:10` 与 `xstars/artifacts.py:47` 定义，已核对）——回滚/升级不丢用户数据；卸载后重装即恢复，`artifacts.py` 的 `SCHEMA_VERSION` 自校验可发现异常工件。
- **开发者模式不受影响**：`docs/macos-developer-setup.md` 流程全程保留（T5.4），安装器回滚后开发者模式仍可用作 fallback。

### 9.4 待决事项（未获批决策，不擅自拍板）

| ID | 事项 | 现状 |
| --- | --- | --- |
| D1 | python-build-standalone **SHA256 具体钉值**与 cpython **3.12.x 具体小版本** | **已裁定（rev 7，实施时取）**：`20260901` / `cpython-3.12.14+20260901`，SHA256 `81a359f1cfadd4da11766534c5913791cea55f26e1bb902cacd2a531bb1e4b2b`，size 24,981,445；已写入 `runtime.lock.json` |
| D2 | `XSTARS.xlam` 是否内置隐藏 `xlwings.conf` sheet 作兜底（researcher 建议双保险，与 V-02/RK-02 相关） | **已裁定（rev 4，2026-09-08）**：内置兜底，`Interpreter` 留空；已落地于已入库的 `XSTARS.xlam` 制品（T0.1） |
| D3 | `XSTARS_mac.xlsm` 是否含示例数据及其内容范围 | **已裁定（rev 6，随交付物）**：用户另存版本保留模板表头与精简示例（Data 32 值、各 Template 45–92 值），已清除 DC2 验收原数据；pytest 仅断言结构不锁定具体值 |
| D4 | 覆盖安装/升级/卸载时是否保留用户 `~/.xstars` 配置与 artifacts | 本 Plan 默认「全程保留」（RK-07 缓解）；如需「卸载时可选清理」需用户确认后加 `--purge-data` 开关 |
| D5 | 制品在仓库的确切路径命名最终确认 | **已裁定（rev 6，随交付物）**：采用 `installer/mac/assets/`，M0 已按此闭合 |
| D6 | CI artifact 保留天数（retention-days） | 默认 GitHub action 上限（90 天），需用户确认 |
| D7 | `XSTARS_mac.xlsm` 投放位置：仅留安装目录（`Templates/`）还是同时复制到 `~/Documents` 等用户可见处 | 影响 T2.2 布局与文档话术 |
| D8 | 构建脚本是否在 CI 使用 uv（而非 pip）加速依赖安装 | 纯实现选择，默认 pip（少一依赖）；不阻塞 |
| D9 | pkg identifier 是否沿用 `com.frank-sysu.xstars`（与旧 xstars-dev 收据同名） | RK-11；需用户确认是否换新 id（如 `com.frank-sysu.xstars.mac`） |
| D10 | 未签名 pkg 是否提供 SHA256 checksum 文件随 Release 分发（完整性自证，非签名） | 文档层面小项，可后补 |
| D11 | T0.1 逐字节断言若因 Excel VBE 在 `vbaProject.bin` 内以 CRLF 存储模块源码而失败（与输入 `.bas` 行尾无关），是否回退为「行尾规范化后逐字节一致」断言（原方案 B） | 实施期 T0.1 首跑即见分晓；失败时制品无需重做，仅断言语义调整，届时凭实测证据请用户确认 → **已裁定（rev 5）：回退为行尾归一化断言**；实测证据：制品提取源 5452B 全 CRLF vs 仓库 5289B 全 LF（差值恰 = 163 = 行数），归一化后逐字节一致，两侧均含 `Attribute VB_Name` 头 |

### 9.5 Git 策略

- **分支**：`feat/macos-installer`（自 `main @ ab30702` 切出）。
- **Draft PR 标题**：`feat: macOS standalone .pkg installer (arm64, user-domain, unsigned)`
- **PR 描述草稿**（可直接使用）：

  > ## Motivation
>
  > macOS 当前仅支持开发者模式（手工 venv + 手工导入 VBA）。本 PR 交付面向终端用户的独立 `.pkg` 安装器：Apple Silicon Mac 用户双击安装，不装 Python、不建 venv、不导 VBA，打开 Excel 即见 XSTARS 功能区；附一等公民卸载脚本。
  >
> ## Approach（依据 plans/20260906-macos-installer.md，rev 1）
>
  > - 构建期下载 python-build-standalone **cpython-3.12 aarch64-apple-darwin install_only_stripped**（固定 SHA256，fail-closed），依赖与 xstars 本体（非 editable）装入其 site-packages，整体入 pkg Payload → `~/Library/Application Support/XSTARS/`。
  > - `installer/mac/build_pkg.py`：装配逻辑 100% 可测 Python，`distribution.xml` 为 arm64-only / min macOS 12.0 / `enable_currentUserHome`（用户级免提权）。
  > - postinstall 四件事：投递 `XSTARS.xlam` → Excel Startup 目录；`xlwings.applescript` → Application Scripts；写 `"INTERPRETER_MAC"` → `~/Library/Containers/com.microsoft.Excel/Data/xlwings.conf`；运行时断言 + 属主修正。
  > - 保留 xlwings `RunPython` 同进程模型：**`ribbon/*.bas` 零修改**（既有 vba-immutability CI 门禁续跑）；不使用 PyInstaller、不产出 `.app`。
  > - 未签名、不公证：文档提供 Sequoia 三途放行指引（「系统设置 → 隐私与安全性 → 仍要打开」/ `xattr -d com.apple.quarantine` / `sudo installer`）。
  > - 顺手修：pyproject 显式声明 Pillow；`xstars/__init__.py` 版本对齐 pyproject（1.1.1）。
  >
> ## Non-goals
>
  > 不签名/公证；不做 universal2；无 `.app`/DMG/Homebrew；不使用 PyInstaller；不改 `ribbon/*.bas`；不支持 WPS for Mac；不修 requirements.txt；不清理缓存目录；运行时 tarball 不入库。
  >
> ## Real-machine acceptance checklist（用户回填）
>
  > - [ ] V-01 Tkinter/ttkbootstrap 开箱可用（`TK_OK`）
  > - [ ] V-02 首次未运行 Excel 时目录预创建有效
  > - [ ] V-03 `INTERPRETER_MAC` 沙盒读取有效（进程路径正确）
  > - [ ] V-04 quarantine 弹窗与「仍要打开」放行路径
  > - [ ] V-05 RunPython 连续 5 次触发稳定
  >
> ## Risks
>
  > Tk init.tcl Quirk（sitecustomize 预设 TCL_LIBRARY 缓解）；Excel 首启覆盖 Containers 配置（仅更新键不覆盖）；跨机分发 Gatekeeper 拦截（文档三途）；制品与 `.bas` 漂移（CI pytest 逐字节断言）。详见 Plan §9.2。

- **PR 拆分决策**：**单 Draft PR**（R11）。依据：7 个 Milestone 全部围绕 `installer/mac/` 单一子系统 + 配套文档/测试，共享同一验收闭环（真机回填），拆分会在 M0 制品入库与 M2 构建依赖间制造跨 PR 阻塞，不满足拆 integration 分支的门槛。
- **合并顺序**：M0 → M1 → M2 → M3 → M4 → M5（可与 M2–M4 并行提交，合并前完成）→ M6，**串行进入同一 PR**（每 Milestone 一组提交，PR 描述内按 Milestone 记录进度）；全部 CI 绿 + V-01~V-05 回填 Pass 后 Draft → Ready for review。

---

## 自查记录（落盘前执行）

1. **9 段齐全且顺序固定**：Goal → Requirements → Non-goals → Research summary → Gap analysis → Milestone 表格 → 分 milestone To-do → Validation contract → 文件级修改范围 + 风险/回滚/待决 + Git 策略。✅ 无遗漏、无调换。
2. **新建路径选择正确**：`plans/20260906-macos-installer.md`，含 Changelog 表格首行 rev 1 / 2026-09-06 / 初稿 / 依据。✅
3. **Milestone 数量**：7 个（M0–M6），≤10 上限、满足 ≤7 目标，无需合并说明。✅
4. **无孤立缺口**：G1→T1.2/T1.3；G2→T2.1/T2.2/T2.3（T1.1 入口）；G3→T2.2/T2.4；G4→T3.1/T3.2；G5→T3.2；G6→T0.1/T0.2/T0.3；G7→T4.1/T4.2；G8→T1.1/T5.1；G9→T5.1；G10→T5.2/T5.3/T5.4/T5.5；G11→T6.1；G12→T2.1/T2.3/T3.1；G13→T1.4/T2.4/T3.3/T4.2（横切，已注明支撑缺口）；G14→T6.2（横切，已注明）；G15→T2.3；G16→T2.1；G17→T2.1。**17 个缺口全部映射到至少一个任务 ID，无孤立任务（每个任务均可回溯缺口或横切约束）。** ✅
5. **每个任务四项齐备**：全部 23 个 To-do（T0.1–T6.2；rev 2 审计计数）均含文件/修改/验收/依赖。✅
6. **Validation contract 可判定**：每项含命令/方式 + 预期 + 通过标准；无法执行项（8.4–8.6 真机）已注明原因与责任人（=用户）。✅
7. **Git 策略齐备**：分支 `feat/macos-installer`、Draft PR 标题、可直接使用的 PR 描述草稿、单 PR 拆分决策与依据、串行合并顺序。✅
8. **歧义全部入待决**：D1–D11 共 11 项（rev 3 增 D11 行尾回退裁决项；含任务指定的 6 项 + RK 相关 4 项）；**已裁定：D1（rev 7，钉值取定）、D2（rev 4）、D3/D5（rev 6，随交付物）、D11（rev 5）**；11 项用户裁定全部可追溯到 Requirements R1–R11；未替用户作任何产品决策（保留 `~/.xstars` 仅为 Plan 默认缓解方案，已标 D4 待确认）。✅
9. **只写入 Plan 文件**：本文件为唯一写入目标（`plans/20260906-macos-installer.md`）；未修改任何源码、测试、配置、脚本或生成物。✅

---

## Review 记录（阶段四）

### 第 1 轮（2026-09-08，三路 fresh-context reviewer 并行）

| Lane | 结果 | 要点 |
| --- | --- | --- |
| 测试与验收覆盖 | 完成 | 14 findings（见下） |
| 正确性与回归 | 完成（fallback 模型续跑） | 1 Blocker + 2 值得修复 + 2 可选 |
| 简洁性与可维护性 | 完成（fallback 模型续跑） | 3 值得修复 + 4 可选；bash LF/路径常数/文档一致性核验干净 |

**F1–F10 修复映射（fix worker 全部落地，验收全绿）**：

| # | 来源 | 修复 | 验证 |
| --- | --- | --- | --- |
| F1 🔴 | coverage | workflow `--prepare-runtime` 与 `--build-pkg` 之间插入 staging import 冒烟（`PYTHONSAFEPATH=1` 防仓库源码遮蔽） | 步骤入 workflow，CI 验证 |
| F2 | correctness+quality | distribution 显式 `enable_localSystem="false" enable_anywhere="false"`；docs 删 `-target /` 变体 | attrib 断言更新 + grep 零命中 |
| F3 | coverage | `BUILT_PACKAGE` 版本化 + 缺包在 darwin 上改 FAIL（不再静默 SKIP） | 单测 |
| F4 | coverage | xlwings 版本不变量对齐制品（AppleScript 文件名版本 vs 内嵌 `xlwings.bas` 版本），删除宿主断言；assets/README 不变量 3 同步更新 | 单测 + README rev 13 |
| F5 | coverage | 真包调 `validate_payload_archive` + `python/bin/python3` 断言 + required 缺失分支 `pytest.raises` 用例 | 单测 |
| F6 | coverage | `--apply` 行为测试（合成 .reg + fake pgrep）：`~/.xstars` 存活、`OPEN` 行删除、备份存在、conf 保留 | 单测（darwin） |
| F7 | quality | uninstall 冲突/重复模式参数 → exit 2 + 回归测试 | 单测 |
| F8 | quality | 双语 README 功能表 "独立安装器（`.exe` / `.pkg`）" | grep |
| F9 | correctness | postinstall 备份共享 xlwings 状态、uninstall 恢复/删除双路径（保护开发者模式共存） | 单测 |
| F10 | correctness | prepare 写 `staging/manifest.json`（xstars/xlwings/lock），build 严格比对不匹配即 fail-closed | 单测 |

**残余清单（可选改进，不在本轮实施）**：D11 归一化半径（`read_text` 连孤立 `\r` 一起吞；编码维度纯 ASCII 下非现存）；R24 零自动校验 + `*.sh` 不在 eol=lf；`Dictionary`/`ThisWorkbook` 模块无锚定（rev 6「三模块一致」声明 2/3 成立）；隐私扫描不覆盖压缩 `vbaProject.bin` 内部；postinstall awk 合并逻辑（生效实现）无行为测试、静态断言宽松项；build_pkg 多条 fail-closed 分支无测试（含 size 不匹配）；Pillow 无防漂移断言；D7 未拍板但被测试锁成双投契约；CI 可缓存 staging/downloads。**RK-01 处置**：staging 实测 Tcl/Tk 资源在位（`tcl9.0/init.tcl` 等）+ F1 冒烟 import ttkbootstrap 间接覆盖，sitecustomize 缓解推迟至 V-01 实测需要时。**验证终态**：安装器单测 43/43；全量 402 passed/13 skipped；pkg ≈137MiB 三层 AppleDouble 零；CI 四项全绿。
