# XSTARS「Run Analysis 支持弹窗内补选区」实施 Plan

- **状态**：rev 1 已批准；M2–M5 代码与 Mac 自动化已实施，M1/M7 真机 spike 和完整手工清单待 Frank
- **计划日期**：2026-10-11
- **基准**：`main` @ `94d938e`（本机 `/Users/frank/Documents/GitHub/xstars`）
- **宿主**：Windows Excel、macOS Excel、Windows WPS（WPS JS 加载项 + 回环 Python 服务）
- **需求来源**：`/Users/frank/xstars-detect-data/brief.md`（2026-10-11）+ 2026-10-11 补充访谈决策 D1–D4（见 §2）
- **验证执行环境**：Mac 本机可自动化（pytest / node / compileall / 白名单 diff）；Windows 与 WPS 真机验证为 Frank 手工清单（M7）

---

## 1. Goal & Non-goals

### 1.1 Goal

点击 Run Analysis 及 WB / qPCR / CCK-8 预设入口时，**不再要求用户事先选中数据区域**：

1. 弹窗顶部 Detected Data 栏：
   - 已选中区域：照常显示各组 `n=`，并显示数据范围地址（如 `Sheet1!$A$1:$D$7`）。
   - 未选中区域：显示「选择数据」入口，用户在弹窗内补选区（输入 A1 地址或手动框选，形态按宿主能力而定）。
2. 两种情况（已选区 / 弹窗内补选区）都能完整走完 分析 → 写回 流程，结果与现状（先选区再弹窗）一致。
3. 无选区时 Preset / Figure / Theme / Export 四页置灰、OK 禁用；补选成功后解锁并刷新组名与 Control / Blank 下拉框。
4. 三条宿主链路全部支持：Windows Excel（InputBox Type=8 框选，spike 决定最终形态）、macOS Excel（A1 地址输入）、WPS（弹窗内经 broker 回调 JS 补选区，长轮询交互通道）。
5. 入口级补选（访谈决策）：ELISA 入口无标准品选区时显式提示补选（D1）；Quick Run 无选区时弹最小补选提示、不新增设置弹窗（D2）。

用户价值：省掉「先回去选区再点按钮」的来回操作；选错区可在弹窗内直接重选，不必取消重来。

### 1.2 Non-goals

- 不改 `ribbon/*.bas`、`ribbon/customUI14.xml`（CI 有 VBA 零 diff 闸门，`.github/workflows/macos-support.yml` `vba-immutability` job）。
- 不修改分析算法、统计逻辑与写回逻辑（写回锚点仍取自「分析所用的选区」，见 R11 说明）。
- 不顺手修先存缺陷（`test_standard_curve` 零浓度回算失败、`test_wps_probe` 挂起）。
- 不扩展到 Standard Curve 与 Transform-only 入口（`_run_standard_curve_impl` main.py:1597、`_run_transform_only_impl` main.py:1869 及 `TransformOnlyDialog` ui_dialog.py:899 不改）。
- 不做 WPS 端任务窗格 UI 重写、不做跨宿主 UI 统一化重构。
- 不引入云服务 / 外部网络调用；WPS 交互通道仅 127.0.0.1 回环并复用既有 token + Origin 校验。

---

## 2. Requirements

来源标注：〔brief〕= brief.md 用户已确认决策；〔访谈〕= 2026-10-11 补充访谈（Frank 拍板）；〔代码〕= 代码库硬约束。

| ID | 需求 | 来源 |
| --- | --- | --- |
| R1 | Run Analysis / WB / qPCR / CCK-8 入口不再要求事先选区；Detected Data 栏已选区显示各组 n + 范围地址，未选区显示「选择数据」入口（输入地址或手动框选） | 〔brief〕目标 |
| R2 | 三宿主（Windows Excel / Mac Excel / WPS）全做 | 〔brief〕决策 1 |
| R3 | 无选区时 Preset / Figure / Theme / Export 置灰；补选成功后解锁并刷新组名与 Control / Blank 下拉框 | 〔brief〕决策 2 |
| R4 | Mac 补选区采用 A1 地址输入（与 ELISA 的 Mac 实现一致，无框选） | 〔brief〕决策 3 |
| R5 | ELISA 入口（Run ELISA）纳入「不要求事先选区」：无有效标准品选区时，入口直接显式提示补选标准品区（Windows InputBox / Mac A1 / WPS promptRange），选完再走现有 ELISADialog 流程；样本区二次选区保持不变 | 〔访谈 D1〕 |
| R6 | Quick Run（run_quick，无设置弹窗）无选区时弹最小补选提示（A1 地址或 InputBox），补选后直接跑，不新增设置弹窗 | 〔访谈 D2〕 |
| R7 | Windows Tk 模态（grab_set）与 InputBox(Type=8) 冲突时：临时 grab_release 恢复框选、改用 A1 地址输入两条降级路径都写入 Plan，由 spike 结果决定 | 〔访谈 D3〕 |
| R8 | WPS 以长轮询交互通道为主方案实现弹窗内补选区；「无选区时先 promptRange 再弹窗」作为降级开关 / 备选预案写入 Plan | 〔访谈 D4〕 |
| R9 | 补选区四场景必须有测试：无选区→补选→正常分析；已选区→范围显示正确；取消补选→干净退出；补选地址非法→友好报错 | 〔brief〕验证标准 |
| R10 | 全量 pytest 与基线对比无新增失败（基线：先存 1 failed + test_wps_probe 须隔离）；`compileall -q xstars tests`；`git diff --check`；本 Plan §7.1 文件白名单 diff 检查；VBA 零 diff | 〔brief〕验证标准、〔代码〕CI |
| R11 | 不改写回逻辑：现状写回锚点取「分析所用选区」（main.py:661/914 处重新读 `book.selection`）；改序后锚点必须取补选后的同一选区，避免用户补选后又在表里点了别处导致输出位置漂移（这是行为对齐，不是写回逻辑变更） | 〔代码〕main.py:658-664、910-916 |
| R12 | WPS 交互通道仅回环：复用 Bearer token（`hmac.compare_digest`）、Origin 白名单、请求字段白名单与 200×200 SelectionPayload 契约校验；不开放任意路径 / 任意字段 | 〔代码〕wps_service.py:426-507、548-647；contracts.py |
| R13 | Tk 对话框必须运行在 worker 主线程；Tk 控件更新不得跨线程直接调用（后台轮询结果须经队列 + `root.after` / `wait_variable` 回到主线程） | 〔代码〕worker.py:169-171 |
| R14 | worker 总超时 300s（`WORKER_TIMEOUT_SECONDS`）约束弹窗内补选等待时长；单次交互等待须有界 | 〔代码〕wps_service.py:34 |

---

## 3. Research summary

### 3.1 brief 行号核实结果（brief 声称 → 实测）

| brief 声称 | 实测（HEAD 94d938e） | 结论 |
| --- | --- | --- |
| ui_dialog.py:52 构造只收 groups/sizes | `SettingsDialog.__init__` ui_dialog.py:52-68 | ✅ 准确 |
| ui_dialog.py:103-108 Detected Data 栏 | banner 构建 ui_dialog.py:102-110 | ✅ 基本准确（端点偏 2 行） |
| main.py:884-905 `_run_impl` | `_run_impl` 为 884-930；「读选区→validate→弹窗」段在 884-908 | ✅ 实质准确（函数终点不符） |
| main.py:600-630 `_run_preset_impl` | `_run_preset_impl` 为 600-664 | ✅ 实质准确（函数终点不符） |
| main.py:424-440 无选区提示只有 Mac | `_entry_selection_or_error` 424-440，`sel is None` 仅 darwin 分支报错 | ✅ 准确 |
| main.py:316 Windows 靠通用异常兜底 | **不符**：316 位于 `_show_error`（310-352）函数体内；实际通用兜底是各入口 wrapper 的 `except Exception`（run 363-364、run_wb 386-387 等）。**且**非 darwin 下 `book.selection` 为 None 时，`_entry_selection_or_error` 返回 None → 入口**静默 return，不报错也不走兜底**；只有 `book.selection` 访问本身抛异常才会走通用兜底。Windows 上 selection 为 None / 选中图表 / 图片的真实行为未实证 → spike S2（T1.2） | ⚠️ 已修正 |
| main.py:1828-1855 Windows InputBox Type=8 | `_select_sample_data` 1828-1855 | ✅ 准确 |
| main.py:1761-1826 Mac Tk 输入 A1 | `_select_sample_data_macos` 1761-1825 | ✅ 准确（端点偏 1 行） |
| wps-addon/spreadsheet.js:105-125 promptRange | `promptRange` ≈107-125（`readSelection` 101-105、`readRange` 67-100） | ✅ 准确 |
| worker.py:180 起弹同一个 SettingsDialog | `_dialog_config` worker.py:174-192，`SettingsDialog` 调用在 ~180-187 | ✅ 准确 |
| 无选区测试 test_macos_support.py:199-213 | `test_darwin_run_entries_fail_friendly_when_picture_selected` ≈203-215 | ✅ 基本准确 |
| ELISA 二次选区测试 564-645、822-854 | A1 输入测试组 ≈550-650；Windows InputBox 契约测试 ≈822-857 | ✅ 基本准确 |
| test_excel_characterization.py:165-177 | `test_second_selection_inputbox_cancel_returns_none` ≈165-183 | ✅ 基本准确 |

行号偏差均为 ±数行，brief 的结构性结论全部成立；唯 main.py:316 一条机制描述错误（见上表）。

### 3.2 代码现状（关键链路，含行级锚点）

**共享对话框（三条链路共用）**
- `SettingsDialog`（xstars/ui_dialog.py:44-896）：`show()` 70-87 创建窗口并 `grab_set()`（84 行）后进入 `mainloop`。Detected Data 栏 102-110 仅拼接 `组名: n=x`，无地址显示、无「选择数据」入口。四个 tab 经 `notebook.add` 593-598 加入且默认选中 Preset（599）； ttk `Notebook` 支持 `notebook.tab(tab_id, state="disabled")` 置灰（待实现验证，见 T2.3）。
- 组名依赖点（置灰/刷新的实现点，brief 风险 4 的核对结论）：Preset 页 Experiment Preset 组合框 353-362；WB Control 下拉 347-352；qPCR Control 下拉 408-417；CCK-8 Control 431-437、Blank 438-447（值列表 `["(none)"] + group_names`）；Figure 页 Comparison Mode `self._control_combo` 216-234、paired 复选框仅当 `len(group_names)==2` 显示 190-193。**WB/qPCR/CCK-8 的下拉当前是局部变量，未存 `self._` 引用**（改刷新必须先存引用）；`_guess_control/_guess_blank` 683-697、`_on_preset_change` 699-712 是补选后需重算的逻辑。`_on_ok` 807-892 直接读各 combo。
- `TransformOnlyDialog`（899-1154）与 `StandardCurveDialog`（xstars/tools/standard_curve_dialog.py）不在本次范围。

**Windows / Mac Excel 链路（xstars/main.py）**
- 入口 wrapper（ribbon VBA `RunPython` 调用，ribbon/ribbon_callbacks.bas:5-25）：`run` 358-366 → `_run_impl` 884-930；`run_wb/qpcr/cck8` 380-410 → `_run_preset_impl` 600-664；`run_quick` 369-377 → `_run_quick_impl` 933-956；`run_elisa` 413-421 → `_run_elisa_impl` 443-589。顺序均为「`_entry_selection_or_error` 读选区（424-440）→ `_read_selection_auto` 读数据（857-881）→ `handler.validate` → `SettingsDialog(groups, sizes).show()`」。
- 无选区现状：Mac 上 `book.selection is None`（如点了图片）→ 友好报错返回（424-440，测试 test_macos_support.py≈203-215）；**Windows 上 `selection` 为 None 时静默返回**（见 3.1 修正项），抛异常才走通用兜底（363-364 等）。
- 补选可复用模式：Windows `InputBox(Type=8)`（`_select_sample_data` 1828-1855，含取消=抛异常返回 None 的契约，测试 test_excel_characterization.py≈165-183、test_macos_support.py≈822-857）；Mac A1 输入 + `_A1_RANGE_RE` 校验（1744-1747）+ 重试循环 + `_silenced_stderr_fd`（1723-1741，xlwings 对非空 stderr 弹失败框的 DC2 N-1 规避）（`_select_sample_data_macos` 1761-1825）。
- ELISA 入口（443-470）：第一步直接把**当前选区**读成标准品数据（`sel.options(pd.DataFrame, header=1, ...)`），无任何提示；样本区二次选区 `_select_sample_data`（494）已有三平台实现。
- 写回锚点：`_run_impl` 912-916 / `_run_preset_impl` 661-664 在写回前**重新读 `book.selection`** 取 row/column —— 改序后必须改用「分析所用选区」的引用（R11）。
- 标签列（WB/qPCR labeled 分支）依赖数据读取结果 `wb_labels`（`_read_selection_auto` 857-881 返回）；弹窗内补选后 labeled 判定须基于补选数据重算，且 `base_config.preset_has_reference` 的预置（`_run_preset_impl` 624-630、`_run_impl` 899-905）也要按补选数据补设。

**WPS 链路（JS → HTTP → broker → worker 子进程）**
- JS 侧（wps-addon/main.js）：`OnAction` ≈266-276 → `runCommand` ≈73-103：对 `SELECTION_COMMANDS`（≈43-51，含 run/run_quick/run_wb/run_qpcr/run_cck8/run_transform_only）**先 `readSelection`（spreadsheet.js:101-105）读当前选区**，读不到（`Application.Selection` 为 null / 非法）→ `readRange` 抛「请先选择一个连续的单元格区域」→ `showError` 弹错，**命令根本发不出去**。`promptRange`（spreadsheet.js≈107-125，InputBox Type=8）目前仅 ELISA（`runElisa` main.js≈106-135）和 Standard Curve 使用。
- broker（xstars/wps_service.py）：`do_POST` 649-701 持 `job_lock` 后同步跑 `SubprocessJobRunner.run`（205-386，`process.wait(timeout=300)` ≈343-354）；`_read_command` 548-647 对全部 `_SELECTION_COMMANDS`（31-40）**硬性要求 `selection` 字段**，缺失即 400 `INVALID_SELECTION`（≈594-607）；请求字段白名单 ≈576-583；`do_OPTIONS` ≈452-465 只放行 `/command`、`/cancel`。服务端是 `ThreadingHTTPServer`（388-404，`daemon_threads=True`）—— `/command` 阻塞期间其他请求可并发处理，这是长轮询交互通道的可行性基础。
- worker（xstars/application/worker.py）：`execute_request` 530-750 对 selection 命令同样硬性要求 `selection` Mapping（≈611-613）；`_DIALOG_COMMANDS = {RUN, WB, QPCR, CCK8}`（330-334）在 ≈563 调 `_dialog_config(selection, config)`（174-192）弹 `SettingsDialog(hide_file_export=True)`；对话框必须主线程（169-171）。worker 是 broker 用 `subprocess.Popen` 拉起的独立进程（wps_service.py:321-339），job 目录内有 request.json / result.json / cancel 文件协议（205-386）—— 交互通道可复用同一目录做文件握手。
- 测试基建：worker 测试经 `execute_request(..., dialog_config=stub)` 注入（tests/test_wps_worker.py:59-74）；broker 测试在 tests/test_wps_service.py；JS 测试用 node `vm` + 假 fetch/Application（wps-addon/tests/main.test.cjs:1-60，`npm test` = `node --test tests/*.test.cjs`，package.json）。
- **WPS 的 ELISA 已天然满足「不要求事先选区」**：`runElisa` 对标准品区和样本区都用 `promptRange` 显式提示（main.js≈106-135）—— R5 在 WPS 侧只需核对与补测试，不需新交互。

### 3.3 外部依据

- Tk `grab_set` 语义：默认为 **local grab**（仅本 Tk 应用内事件重定向），理论上不拦截发给其他进程（Excel InputBox 原生窗口）的鼠标事件；但 Tk 窗口层级 / 焦点抢占与 Excel 模态 InputBox 的叠加行为无公开可引用的结论，且本仓库从未在 `grab_set` 生效期间调用过 `InputBox`（ELISA 流程中两者从不并存）。置信度：低 —— 必须 spike（T1.1）。未做外部网页调研（离线环境），如实施者可联网可补充 tkinter 官方 grab 文档链接。
- VBA→Python 同步调用期间 Excel UI 状态：`RunPython` 由 VBA 同步等待（ribbon .bas），期间 Excel 主线程被占用；`InputBox(Type=8)` 之所以在现有 ELISA 流程可用，是因其由 Excel 自身模态展示并泵消息循环 —— 该推断来自现有已验证功能（WPS 计划 M0.3/M0.4 与 ELISA 特性），非新证据。置信度：中。
- WPS 内嵌浏览器并发 fetch（长轮询与 /command 并行）：**未验证**（历史仅发现 `window.prompt` 不可用，wps 计划 changelog）。置信度：低 —— spike S3（T1.3）。

### 3.4 推荐方案与被否决方案

**A. SettingsDialog 弹窗内补选区（三宿主共用骨架）**
- 推荐：`SettingsDialog` 新增可选参数 `selection_label`（已选区地址）与 `request_selection` 回调（宿主注入的补选 provider）；新增内部状态机 `_apply_data(groups, sizes, label, has_label_column)` 负责横幅更新、四页解锁/置灰、下拉刷新；补选结果经 `dialog.selected_payload` 交还调用方。调用方（main.py / worker.py）把「读选区→validate→弹窗」改为「弹窗（含可选 provider）→（provider 内完成补选+validate+重试）→ 用 selected_payload 走既有分析分支」。
- 否决：把补选做成独立于 SettingsDialog 的前置小弹窗（不满足 brief「弹窗内」要求，且与 D4 降级开关重复）；把数据读取搬进 ui_dialog.py（宿主耦合进共享 UI，破坏 worker 无 COM 依赖的结构）。

**B. Windows 补选交互（spike 决定主形态，R7）**
- 候选 B1（主）：「选择数据」回调内先 `grab_release()` → 调 `book.app.api.InputBox(Type=8)` → 成功后 `grab_set()` 恢复，保住框选交互。
- 候选 B2（降级）：回调内不碰 Excel，弹 `simpledialog.askstring(parent=self._root)` 输入 A1 地址（复用 Mac helper 的校验/重试模式），放弃框选。
- spike S1 矩阵：{grab 保持 / 释放} × {Tk 窗口 topmost 与否} × {正常选区 / 取消 / 多区域选区}，任何一格满足「InputBox 可见、可在 Excel 里框选、回调正确返回地址、弹窗恢复可用」即记录为可行组合。
- 否决：后台线程轮询 `book.selection` 等用户改选（VBA 同步等待期间 Excel UI 不接受输入，除非经 Excel 自身模态泵消息；且跨线程 COM 调用有套间问题）。

**C. Mac 补选交互**
- 推荐：复用 `_select_sample_data_macos` 模式抽出通用 `_prompt_range_macos(book, sheet, title, message)`（A1 正则校验 + 读取 + 友好报错重试 + `_silenced_stderr_fd`）；弹窗内版本以 `simpledialog.askstring(parent=dialog_root)` 嵌入同一 Tk 会话。嵌套对话框在 grab 下的表现需 Mac 本机验证（T4.2 验收含此点，异常则改独立 Toplevel）。
- 否决：引入 Mac 原生框选（AppScript 无可靠选区监听接口，超范围）。

**D. WPS 弹窗内补选区（长轮询交互通道，R8 主方案）**
- 推荐：**job 目录文件握手 + broker 长轮询端点**，全程复用既有安全边界：
  1. worker 弹 `SettingsDialog(selection=None, request_selection=provider)`；provider 在「选择数据」回调里向 job 目录写 `interaction_request.json`（`{id, prompt}`），后台线程轮询 `interaction_response.json`（单次等待 ≤120s，期间检查 cancelPath），结果经队列 + `root.after`/`wait_variable` 回主线程更新弹窗（R13）。
  2. broker `SubprocessJobRunner.run` 把 `process.wait(timeout)` 改为分片循环（如 0.2s 步进，总时长不变），监视 `interaction_request.json` 出现即登记到 server 的单槽 `pending_interaction`（ThreadingHTTPServer 保证 `/command` 阻塞期间可服务其他请求）。
  3. 新端点：`GET /interaction`（已鉴权，长轮询 ≤30s：有 pending 请求立即返回，否则挂起至超时返回 `{none}`）；`POST /interaction`（已鉴权 + Origin 校验 + 字段白名单，响应体按 `SelectionPayload.from_dict` 契约校验后原子写入活动 job 目录的 `interaction_response.json`）。`do_OPTIONS` 放行新路径。
  4. JS：`runCommand` 在无有效选区时（`readSelection` 失败改为不报错）发送**不带 selection** 的命令，并在 `await /command` 的同时并发发起 `GET /interaction` 长轮询；收到请求 → `promptRange` → `POST /interaction` → 继续等 `/command`。
  5. 降级开关（D4）：JS 收到 `GET /interaction` 404（旧服务）或 `POST /command` 返回 `INVALID_SELECTION`（旧 broker 硬性要求 selection）→ 退回「先 `promptRange` 补选、带 selection 重发命令」（补选发生在弹窗外，但同样满足「不需要事先选区」）。是否默认启用长轮询由加载项 config（`config.template.js` 注入字段）控制，载体待实施确认（见待决事项 3）。
- 否决：worker 直连 JS（worker 子进程与 WPS 页面无任何通道，物理上不通）；WebSocket / SSE（引新依赖且超出回环 HTTP 最小面）；broker 主动 push（JS 侧 fetch 无服务端推送语义，长轮询是等价最小实现）；把弹窗搬到 JS 侧重写（Non-goal）。

**E. Quick Run（D2）**：Windows InputBox Type=8 / Mac A1 输入 / WPS `promptRange`（JS 侧在 `readSelection` 失败时调用）取得选区后直接走现有 `_run_quick_impl` / `run_quick` 命令，不新增设置弹窗。

**F. ELISA（D1）**：Windows/Mac 的 `_run_elisa_impl` 把「读当前选区」改为「无有效选区时用 B/C 的补选 helper 显式提示标准品区」（读取仍 `header=1` 同构）；样本区二次选区不动。WPS 侧已满足（3.2），仅核对 + 补测试。

### 3.5 证据缺口（未读到 / 未验证，不作为已知事实）

1. Windows 上 `book.selection` 在「无有效数据选区 / 选中图表 / 选中图片 / Ctrl 多区域」下的真实返回（None / Shape / Range / 抛异常）—— 未实证（本机无 Windows Excel），T1.2 spike。
2. Tk `grab_set` 与 Excel `InputBox(Type=8)` 并存的焦点/交互行为 —— 无代码证据，T1.1 spike。
3. WPS 内嵌浏览器在 `/command` 挂起期间并发长轮询 fetch 的行为 —— 未验证，T1.3 spike。
4. ttk `Notebook.tab(state="disabled")` 在 ttkbootstrap 主题下的实际置灰效果 —— API 存在但本仓库未用过，T2.3 验收时确认（若无效，退化为「进入即弹补选提示 + 隐藏 tab」方案，见风险 R-3）。
5. 全量 pytest 基线的确切失败用例 ID（brief 只说「先存 1 failed」）—— 计划阶段未执行测试（只读环境无 bash），T6.1 实施时先在 main 上记录基线快照。
6. 本 Plan 中 wps-addon/main.js、service-client.js 的行号为「≈」（文件读取无行号标注），以函数名为准；实施者首步可用 `grep -n` 校正。

---

## 4. Gap analysis

| ID | 功能缺口 | 现状与影响 | 补齐任务 |
| --- | --- | --- | --- |
| G1 | SettingsDialog 无法在无数据状态下打开并自行获取数据 | 构造必须收 groups/sizes（ui_dialog.py:52-68）；banner 只显示 n（102-110） | T2.1, T2.2, T2.5 |
| G2 | 无选区时四页未置灰、补选后下拉不刷新 | 全部控件按已有 group_names 构建（216-234、347-352、408-417、431-447），WB/qPCR/CCK-8 下拉未存 self 引用 | T2.3, T2.4 |
| G3 | Excel 链路流程为「读选区→validate→弹窗」，无选区即终止 | main.py 884-930 / 600-664；Windows 静默退出、Mac 报错返回 | T3.2, T4.2（写回锚点对齐 T3.2/T4.2 内含） |
| G4 | Windows 弹窗内框选交互（Tk 模态 × InputBox）不存在 | InputBox 仅在无 Tk 模态并存的 ELISA 流程用过（1828-1855） | T1.1, T1.2, T3.1 |
| G5 | Mac 补选 helper 与 ELISA 样本选区耦合，不可复用 | `_select_sample_data_macos`（1761-1825）写死 ELISA 文案与 OD 语义 | T4.1 |
| G6 | Quick Run 无选区直接失败（Windows 静默 / Mac 报错 / WPS 弹错） | `_run_quick_impl` 933-956；wps main.js `readSelection` 先行失败 | T3.3, T4.3, T5.5 |
| G7 | ELISA 入口（Win/Mac）要求事先选好标准品区 | `_run_elisa_impl` 443-470 直接读当前选区；WPS 已满足（main.js runElisa） | T3.4, T4.3, T5.6 |
| G8 | WPS 三层（JS / broker / worker）硬性要求选区，命令发不出 | spreadsheet.js:101-105、wps_service.py:594-607、worker.py:611-613 | T5.1, T5.2, T5.3, T5.5 |
| G9 | worker↔broker↔JS 之间没有任何交互通道 | broker 仅 /health /command /cancel；worker 仅文件协议收请求发结果 | T5.1, T5.2, T5.4 |
| G10 | 新行为的回归、静态检查与真机验证未定义 | 现有测试只覆盖「无选区报错（Mac）」与 ELISA 二次选区 | T2.5, T3.6, T4.4, T5.7, T6.1-T6.4, T7.1-T7.3 |

自查：无孤立缺口（G1–G10 均映射到 ≥1 任务；横切任务 T1.x 支撑 G4/G8/G9 的可行性，T2.x 支撑 G1–G3，T6.x/T7.x 支撑 G10）。

---

## 5. Milestones 与任务

| Milestone | Status | Dependencies | Validation |
| --- | --- | --- | --- |
| M1 Windows/WPS 可行性 spike | [ ] | — | S1–S3 清单每项有结论（可行/不可行/触发降级）；结论与选型回填 PR；无源码改动（spike 脚本仅入 `poc/windows-selection-spike/` 白名单或不提交） |
| M2 SettingsDialog 弹窗内补选区 UI | [x] | M1（provider 交互形态确定后冻结 T2.1 接口） | 新单测全绿；`python -m pytest tests/test_macos_support.py tests/test_excel_characterization.py -q` 无回归；`python -m compileall -q xstars tests` |
| M3 Windows Excel 链路 | [x] | M1, M2 | mock 单测（补选成功/取消/非法/锚点/ELISA/Quick）全绿；真机行为留待 M7 |
| M4 Mac Excel 链路 | [~] | M2 | 单测全绿 + 本机 Mac Excel 真机检查单（§6.3）全绿 |
| M5 WPS 交互通道与降级 | [x] | M1（S3 结论）, M2 | `cd wps-addon && npm test && npm run check` 全绿；broker/worker 新测试全绿；降级路径有测试 |
| M6 回归与收尾 | [~] | M2–M5 | 全量 pytest 对比基线无新增失败；compileall / git diff --check / §7.1 白名单 diff / VBA 零 diff 全绿 |
| M7 Windows/WPS 真机验证 | [ ] | M6 | Frank 手工清单（§6.4）全绿，结果回填本文件并勾选任务 |

### M1 — Windows/WPS 可行性 spike（只调研，不改产品代码）

- [ ] T1.1 Windows spike S1：Tk grab × InputBox(Type=8) 焦点/交互矩阵
  - 文件：spike 脚本（可选提交 `poc/windows-selection-spike/spike_tk_inputbox.py`，或仅本地临时文件不提交）；结论回填 PR 评论与本 Plan §3.4 B
  - 做法：本地一次性工作簿 + 一段 `RunPython "import spike_tk_inputbox; ..."` VBA（本地工作簿 VBA，不触碰仓库 `ribbon/`）；脚本连 `xw.Book.caller()` 后弹 `grab_set` 的 Tk 窗口，按钮回调按 §3.4-B 矩阵调 `InputBox(Type=8)`，记录：InputBox 是否可见、能否在 Excel 框选、回调返回地址、取消行为、弹窗是否恢复 grab、有无死锁
  - 验收：矩阵 4 格（grab 保持/释放 × topmost 开/关）各有「可行/不可行」结论；至少一格可行，或确认全部不可行并启用 B2（A1 输入）
  - 依赖：Frank 在 Windows 真机执行
- [ ] T1.2 Windows spike S2：`book.selection` 状态矩阵
  - 文件：同上 spike 脚本；结论回填 §3.1 修正项与 T3.5 判定谓词
  - 做法：同一真机会话中分别记录「活动单元格正常选区 / 选中图表 / 选中图片 / Ctrl 多区域 / 无 anything」时 `book.selection` 的返回类型或异常
  - 验收：得到「Windows 有效数据选区」判定谓词（返回 Range 且 Areas.Count==1 且在 worksheet 上？），写进 T3.5 验收
  - 依赖：Frank；与 T1.1 同会话执行
- [ ] T1.3 WPS spike S3：内嵌浏览器并发长轮询可行性
  - 文件：结论回填 PR（可用临时调试版加载项或现有 `promptRange` + 手工 fetch 试验，不改仓库产品代码）
  - 做法：在 WPS 内嵌页面发一个挂起的 `/command`（如长任务）同时并发 `fetch GET /health` 循环，确认两个请求能并行完成、无连接数/引擎限制；并确认长轮询 30s 挂起不冻结 Ribbon/页面
  - 验收：并发 fetch 可行/不可行结论；不可行则 D4 降级开关升级为主方案（须经 Frank 确认）
  - 依赖：Frank 在 Windows + WPS 真机执行
- [ ] T1.4 spike 结论固化
  - 文件：`plans/20261011-run-analysis-in-dialog-selection.md`（本文件，增补 rev 记录）
  - 验收：Windows 主形态（B1 或 B2）与 WPS 主/降级关系写入 Changelog；后续 Milestone 的对应任务按结论执行
  - 依赖：T1.1–T1.3

### M2 — SettingsDialog 弹窗内补选区 UI 与状态机（Mac 可开发可测）

- [x] T2.1 构造接口扩展（向后兼容）
  - 文件：`xstars/ui_dialog.py`（`__init__` 52-68）
  - 内容：新增关键字参数 `selection_label: str | None = None`、`request_selection: Callable[[], SelectionSummary | None] | None = None`；`SelectionSummary` 为轻量数据类（`group_names`、`group_sizes`、`label`（地址）、`has_label_column: bool`、`payload`（调用方专用的不透明对象，如 (wb_labels, df_wide) 或 SelectionPayload））；`groups/sizes` 允许为空表示初始无选区；新增 `self.selected_payload` 供 show() 后调用方读取
  - 验收：现有构造调用（main.py、worker.py、全部测试）不改仍可用；新参数默认值下行为与现状逐位一致（既有测试全绿）
- [x] T2.2 Detected Data 栏改造
  - 文件：`xstars/ui_dialog.py`（banner 102-110）
  - 内容：有数据时显示 `组: n | … · 范围 label`；无数据时显示「未检测到数据选区」+「选择数据…」按钮（调 `request_selection`，返回 None 保持无数据并给出提示文案）；按钮在 `request_selection is None` 时不渲染（旧行为）
  - 验收：单测覆盖两种渲染态与按钮回调（mock provider）；Mac 手动可见性检查
- [x] T2.3 无数据置灰
  - 文件：`xstars/ui_dialog.py`（notebook 593-603、OK 按钮 606-620）
  - 内容：无数据时 `notebook.tab(t, state="disabled")` 四页全灰、OK `state="disabled"`；有数据恢复 `normal`。若 ttkbootstrap 主题下 disabled tab 无视觉效果（证据缺口 4），退化为：无数据时仅显示一个「请先选择数据」帧（隐藏/禁用四页），并在 PR 记录
  - 验收：置灰/恢复有单测（fake root 或纯状态函数）；无法单测的视觉部分进 Mac 手动检查单
- [x] T2.4 补选成功后解锁与刷新
  - 文件：`xstars/ui_dialog.py`（216-234、347-352、408-417、431-447、683-697、699-712、190-193）
  - 内容：新增 `_apply_data(summary)`：更新 banner；启用四页与 OK；把 WB/qPCR/CCK-8 的 Control/Blank 下拉改为存 `self._` 引用并重设 `values=summary.group_names`（Blank 为 `["(none)"]+…`）；重跑 `_guess_control/_guess_blank` 设默认值；Comparison Mode `_control_combo` 重设 values；paired 复选框按 `len(groups)==2` 显隐；`_on_preset_change()` 重算；`has_label_column` 为真时把 WB/qPCR `_wb_ref_var/_qpcr_ref_var` 默认设 True（对齐 main.py:624-630/899-905 的预置语义）
  - 验收：单测：空 → apply → 断言各下拉 values/默认值/置灰状态；连续两次 apply（补选后重选）幂等
- [x] T2.5 可测试性抽取与新单测
  - 文件：`xstars/ui_dialog.py`（把「置灰判定/横幅文案/下拉值列表」抽为不依赖 Tk 的纯函数或独立方法）；新建 `tests/test_ui_dialog_selection.py`
  - 验收：不创建真实 Tk 即可测状态逻辑（沿用 test_macos_support.py 的 `_install_tk_modules` mock 模式或纯函数）；覆盖 R9 四场景中「已选区范围显示」「取消补选保持置灰」的 UI 侧断言
- [x] T2.6 明确不改：`TransformOnlyDialog`、`StandardCurveDialog`、`ELISADialog` 构造签名不动（ELISADialog 无需弹窗内补选，见 R5/D1）
  - 文件：（无）
  - 验收：`git diff --name-only` 不含上述三个类的文件改动（xstars/tools/standard_curve_dialog.py、xstars/presets/elisa_dialog.py 不在白名单）

### M3 — Windows Excel 链路（代码在 Mac 上以 mock 开发，真机验证留 M7）

- [x] T3.1 通用 Windows 补选 helper
  - 文件：`xstars/main.py`（参照 1828-1855 模式新增 `_prompt_range_windows(book, sheet, title, message)`）
  - 内容：`InputBox(Type=8)` → 校验返回 Range 的单区域性与所在 sheet（多区域/跨表→友好报错重试或返回 None，与 T1.2 谓词一致）→ 返回 (range, raw_df)；按 T1.1 结论决定是否在调用点 `grab_release/grab_set`（若 B2，则 helper 不调 InputBox 而走 T4.1 的 A1 输入）
  - 验收：mock 单测：正常返回/取消（异常→None）/多区域拒绝/地址非法报错文案（沿用 `_friendly_message` 风格）
- [x] T3.2 `_run_impl` / `_run_preset_impl` 流程改序
  - 文件：`xstars/main.py`（884-930、600-664）
  - 内容：选区有效 → 现状路径（数据预读，`selection_label=sel.address` 类字段）；无选区 → `SettingsDialog(request_selection=provider)`（provider 内完成 InputBox/读数据/`_read_selection_auto`/`handler.validate`/重试，返回 SelectionSummary）；`show()` 后统一从 `dialog.selected_payload` 取 (wb_labels, df_wide, sheet, 选区引用)；labeled 分支判定、`base_config` 预置补设（has_reference 等）改为对最终数据生效；**写回锚点改用分析用选区引用**（不再重读 `book.selection`，R11）；取消（config None / 无数据）→ 干净 return
  - 验收：mock 单测：①无选区+provider 成功→分析与写回锚点用补选区；②无选区+provider 取消→无任何写回/弹错；③已选区→行为与现状 golden 一致（对照改前断言）；④补选后 labeled WB 分支可用（provider 返回带标签列数据）
- [x] T3.3 Quick Run 最小补选（D2）
  - 文件：`xstars/main.py`（`_run_quick_impl` 933-956）
  - 内容：无有效选区 → 调 T3.1 helper（B2 路径则 A1）→ 成功后按补选区走现状流程，失败/取消 → 友好报错返回
  - 验收：mock 单测：补选成功出图；取消无副作用
- [x] T3.4 ELISA 入口显式提示补选（D1）
  - 文件：`xstars/main.py`（`_run_elisa_impl` 443-470）
  - 内容：无有效选区 → T3.1 helper 提示「选择标准品区域（首行浓度表头）」→ 读入后走现有 458-470 起的同构处理；样本区 `_select_sample_data`（494）不动
  - 验收：mock 单测：补选标准品→fit 正常推进；取消→干净退出
- [x] T3.5 Windows 有效选区判定谓词（依 T1.2 结论）
  - 文件：`xstars/main.py`（重构 `_entry_selection_or_error` 424-440 为 `_read_valid_selection(book)`，三平台共用判定：None/非 Range/多区域 → 视为无选区而非异常）
  - 验收：单测（mock）覆盖谓词各分支；Mac 既有无选区测试（test_macos_support.py≈203-215）改走新谓词仍绿
- [x] T3.6 Windows 链路单测
  - 文件：`tests/test_excel_characterization.py`（追加）、必要时 `tests/test_macos_support.py`（Windows 契约段≈822-857 风格）
  - 验收：R9 四场景在 Windows mock 路径全部有断言；`python -m compileall -q xstars tests` 通过

### M4 — Mac Excel 链路（本机可完整验证）

- [x] T4.1 通用 Mac 补选 helper
  - 文件：`xstars/main.py`（把 `_select_sample_data_macos` 1761-1825 抽象为 `_prompt_range_macos(book, sheet, title, message, parent_root=None)`；ELISA 调用点改传自己的文案/语义，行为不变）
  - 验收：既有 ELISA A1 测试（test_macos_support.py≈550-650）不改断言全绿；新 helper 支持传入 dialog root 作 parent
- [x] T4.2 Mac 弹窗内补选接入
  - 文件：`xstars/main.py`（`_run_impl`/`_run_preset_impl` Mac 分支）、`xstars/ui_dialog.py`（如需嵌套 dialog 支撑）
  - 内容：provider = `_prompt_range_macos(..., parent_root=dialog_root)` + `_silenced_stderr_fd` + 校验重试；确认 `simpledialog.askstring(parent=弹窗root)` 在 `grab_set` 下可用（证据缺口外的 Mac 本机可验点），异常则改独立 Toplevel 并记录
  - 验收：mock 单测 + Mac 真机检查单（§6.3）第 1–4 项
- [x] T4.3 Mac Quick Run / ELISA 入口
  - 文件：`xstars/main.py`（`_run_quick_impl`、`_run_elisa_impl` Mac 路径）
  - 内容：Quick 用 A1 输入；ELISA 标准品区无选区时 A1 输入
  - 验收：mock 单测 + §6.3 第 5–7 项
- [ ] T4.4 Mac 测试与真机验证
  - 文件：`tests/test_macos_support.py`（追加无选区→补选场景）
  - 验收：`python -m pytest tests/test_macos_support.py -q` 全绿；§6.3 检查单全绿并留截图/记录回填 PR

### M5 — WPS 长轮询交互通道与降级（§3.4-D 设计）

- [x] T5.1 worker 侧：允许无 selection + 交互文件协议
  - 文件：`xstars/application/worker.py`（execute_request ≈611-613、`_dialog_config` 174-192）、`xstars/application/contracts.py`（如需交互 DTO）
  - 内容：`_DIALOG_COMMANDS` 允许请求不带 selection（`selection=None` 传入 `dialog_config`）；新增 job 目录交互函数：`request_interaction(job_dir, prompt)`（原子写 request.json）、`wait_interaction(job_dir, timeout, cancel_path)`（轮询 response.json）；`_dialog_config` 在 `selection is None` 时构造带 provider 的 `SettingsDialog`，provider 后台线程等待 + 队列 + `root.after`/`wait_variable` 回主线程（R13）；取消/超时→provider 返回 None（弹窗保持无数据态，用户可重试或 Cancel）
  - 验收：单测（mock SettingsDialog + mock 交互函数）：无 selection 请求不再抛 `INVALID_SELECTION`；provider 成功/取消/超时三分支；`WorkerCancelled` 语义不变（dialog Cancel → 取消整个 job）
- [x] T5.2 broker 侧：监视 + 长轮询端点
  - 文件：`xstars/wps_service.py`（`SubprocessJobRunner.run` 205-386、`do_GET`/`do_OPTIONS`/`do_POST` 452-701）
  - 内容：`run` 的 `process.wait` 改分片循环并监视 `interaction_request.json` → 登记到 server 单槽 pending（挂到 runner/server 共享状态）；`GET /interaction`（鉴权 + 长轮询 ≤30s）；`POST /interaction`（鉴权 + Origin + 字段白名单 + `SelectionPayload.from_dict` 契约校验 → 原子写入活动 job 目录）；`do_OPTIONS` 放行 `/interaction`；job 结束清理 pending（finally 段 386 附近）；超时/取消后端点返回确定性错误
  - 验收：tests/test_wps_service.py 新增：无 pending 时 GET 立即/超时返回；POST 校验失败 400；写文件成功后 worker 可读；/command 挂起期间并发 GET 可服务（复用 ThreadingHTTPServer 测试手法）；鉴权失败 401
- [x] T5.3 worker `execute_request` 数据流接通
  - 文件：`xstars/application/worker.py`（`_dialog_config`、execute_request 后续分析段）
  - 内容：provider 返回 SelectionPayload 后：`split_selection_labels` → `SelectionSummary`（group 名/大小/address/has_label_column）→ 弹窗 `_apply_data`；`show()` 后用 `selected_payload` 替代原 `selection` 走 `_configure_preset` / `analyze_selection` / `_output_start_cell`（锚点对齐 R11）
  - 验收：单测：无 selection 请求 + 假 provider 数据 → 输出 writebackPlan 与带同数据的 selection 请求逐字段一致（golden 对照）
- [x] T5.4 JS 客户端：并发长轮询与交互响应
  - 文件：`wps-addon/service-client.js`（`command` ≈170-233 附近扩展）
  - 内容：`command()` 发起 `/command` 的同时（仅当命令属弹窗类且未携带 selection）并发 `GET /interaction` 长轮询；收到请求 → 调注入的 `onInteraction` 回调（由 main.js 提供 `promptRange`）→ `POST /interaction` 提交选区；`abort` 时中止长轮询；404（旧服务）→ 标记交互不可用
  - 验收：`cd wps-addon && npm test` 新用例：并发两 fetch、promptRange→POST 顺序、取消/abort、404 降级标记；`npm run check` 通过
- [x] T5.5 main.js：无选区不发错 + 降级开关
  - 文件：`wps-addon/main.js`（`runCommand` ≈73-103）、`wps-addon/config.template.js`（开关字段，载体见待决事项 3）
  - 内容：`SELECTION_COMMANDS` 中弹窗类命令：`readSelection` 失败 → 静默改发无 selection 请求（走 T5.4 通道）；`run_quick`：readSelection 失败 → `promptRange` 预补选后发命令（D2，弹窗外补选属可接受——本就无弹窗）；降级开关启用或通道不可用 → 弹窗类命令退回「promptRange 预补选→带 selection 重发」（D4）；`/command` 返回 `INVALID_SELECTION`（旧 broker）同样触发重试
  - 验收：node 测试：无选区不弹错、降级路径、quick 预补选、旧服务兼容重试；`npm test && npm run check` 全绿
- [x] T5.6 WPS ELISA 核对（不改行为）
  - 文件：`wps-addon/tests/main.test.cjs`（补断言）
  - 内容：确认 `runElisa` 已双 promptRange 满足 R5，无需改动；补一条「ELISA 无选区也可完成」的回归用例
  - 验收：用例绿；`git diff` 显示 main.js 的 runElisa 无行为改动
- [x] T5.7 端到端联调（Mac 本机可跑 broker+worker+假 JS）
  - 文件：（不改产品代码；联调用 tests/ 或本地脚本）
  - 内容：用 tests/test_wps_service.py + test_wps_worker.py 的假 fetch/假 Application 串一条「无 selection → 交互 → 出 plan」的链路
  - 验收：链路测试绿；真机 WPS 留 M7

### M6 — 回归与收尾

- [x] T6.1 基线对比
  - 文件：（不改源码；记录到 PR）
  - 内容：分支起点先在 main 上跑 `python -m pytest -q --ignore=tests/test_wps_probe.py` 记录基线失败集（brief 称先存 1 failed；确切 ID 为证据缺口 5）；M6 在分支同命令重跑，新增失败=0；`test_wps_probe` 持续隔离
  - 验收：基线快照与最终结果贴 PR；差集为空
- [x] T6.2 静态检查
  - 文件：（不改）
  - 内容：`python -m compileall -q xstars tests`；`git diff --check`；（如 ruff 配置存在则 `ruff check xstars tests`——计划阶段未确认配置，存在才跑）
  - 验收：全绿
- [x] T6.3 白名单与 VBA 零 diff
  - 文件：（不改）
  - 内容：`git diff --name-only origin/main...HEAD` 输出 ⊆ §7.1 白名单；`git diff --exit-code origin/main...HEAD -- 'ribbon/*.bas'`
  - 验收：两条命令退出码 0
- [ ] T6.4 Mac 全量验证
  - 内容：`python -m pytest -q --ignore=tests/test_wps_probe.py`；`cd wps-addon && npm test && npm run check`；§6.3 真机检查单复核
  - 验收：全绿；记录回填

### M7 — Windows / WPS 真机手工验证（责任人：Frank）

- [ ] T7.1 Windows Excel 清单（§6.4-A）
  - 文件：结果回填本文件
  - 验收：清单全部「预期结果」达成；任何不符记录为缺陷并回到对应 Milestone
- [ ] T7.2 WPS 清单（§6.4-B）
  - 文件：结果回填本文件
  - 验收：同上；含降级开关两种模式各验一遍
- [ ] T7.3 收尾
  - 验收：T7.1/T7.2 证据齐；本文件全部任务勾选；PR 转化决策交 Frank

---

## 6. Validation contract

### 6.1 自动化（Mac 本机可执行，实施者每 Milestone 末运行）

| 检查项 | 命令 / 方式 | 预期 | 通过标准 |
| --- | --- | --- | --- |
| 目标行为单测（R9 四场景） | `python -m pytest tests/test_ui_dialog_selection.py tests/test_macos_support.py tests/test_excel_characterization.py tests/test_wps_worker.py tests/test_wps_service.py -q` | 新增用例全绿 | 0 failed；四场景各有 ≥1 用例断言 |
| 全量回归 | `python -m pytest -q --ignore=tests/test_wps_probe.py`（M6 与基线差集比对） | 与 main 基线失败集一致 | 新增失败 = 0 |
| WPS 加载项 | `cd wps-addon && npm test && npm run check` | 全绿 | 0 failing node test；语法检查通过 |
| 编译 | `python -m compileall -q xstars tests` | 无输出 | 退出码 0 |
| 空白/冲突检查 | `git diff --check` | 无输出 | 退出码 0 |
| 白名单 diff（对应旧计划 §9.1 语义） | `git diff --name-only origin/main...HEAD` | 输出 ⊆ §7.1 | 无白名单外文件 |
| VBA 零 diff | `git diff --exit-code origin/main...HEAD -- 'ribbon/*.bas'` | 无 diff | 退出码 0 |
| 无法本机执行项 | Windows COM / WPS 真机行为（T1.1–T1.3、§6.4） | — | 责任人 Frank，按 M1/M7 清单执行并回填 |

### 6.2 自动化用例必须覆盖的行为矩阵

1. 无选区 → 弹窗打开即置灰 → 补选（Windows mock InputBox / Mac mock A1 / WPS 假交互文件）→ 解锁+下拉刷新 → 分析写回成功，锚点=补选区。
2. 已选区 → 横幅含范围地址与各组 n → 行为与改前 golden 一致。
3. 补选取消（InputBox cancel / A1 取消 / 交互超时）→ 弹窗保持无数据态或干净退出，无写回、无异常弹栈。
4. 补选地址非法 / 多区域 / 跨表 → 友好报错并允许重试（沿用 `_friendly_message`/`_show_error` 文案风格）。
5. Quick Run 无选区 → 最小提示 → 成功出图；取消无副作用。
6. ELISA 无标准品选区 → 显式提示 → 走通 fit→样本→回算。
7. WPS：无 selection 请求合法；交互响应契约校验失败 400；鉴权失败 401；旧服务兼容降级；`/command` 挂起期间并发 GET 可服务。
8. 取消语义：worker Cancel、JS `cancelActiveRequest`、broker `/cancel` 在交互等待期间均能终止 job（cancelPath 检查）。

### 6.3 Mac Excel 真机检查单（本机可执行，实施者执行）

前置：安装版 XSTARS（参照 `macos-pkg-acceptance-runbook.md`）+ `XSTARS_Templates.xlsx`。每项记录结果（截图或文字）回填 PR。

1. 不选任何区域，点 Run Analysis：弹窗打开，四页置灰、OK 禁用、Detected Data 显示「选择数据」入口。
2. 点「选择数据」，输入 `Data!A1:D7` 类合法 A1（模板 Data sheet）：四页解锁、下拉出现组名、横幅显示范围地址；OK 后图表/统计表正常生成且位置相对补选区。
3. 输入非法地址（如 `named_range`、跨 sheet 引用）：弹友好错误，可重试；点取消：回到无数据态，可再次发起。
4. 先选好区再点 Run Analysis：横幅显示 n + 地址，行为与旧版一致。
5. 不选区点 Quick Run：弹 A1 输入；补选后直接出图；取消无副作用。
6. 不选区点 ELISA：提示选标准品区（A1 输入）→ 后续样本选择与回算正常。
7. WB / qPCR / CCK-8 preset 入口重复 1–3 的无选区→补选流程（含带标签列数据走 labeled 分支）。

### 6.4 Windows / WPS 真机手工清单（责任人：Frank）

**A. Windows Excel**（在 M1 spike 环境复用）：

| # | 步骤 | 预期 |
| --- | --- | --- |
| A1 | 不选区点 Run Analysis（含 WB/qPCR/CCK-8） | 弹窗打开、四页置灰、可见「选择数据」；不报错不静默退出 |
| A2 | 点「选择数据」后按 spike 选定形态（B1 框选 / B2 A1）补选 | InputBox 可交互（B1）或 A1 输入生效；弹窗恢复焦点与 grab；解锁+下拉刷新 |
| A3 | 补选后 OK | 分析写回正常；输出锚点在补选区旁 |
| A4 | 补选中点 InputBox Cancel | 弹窗回到无数据态，可重试；无死锁 |
| A5 | 已选区正常路径 | 横幅含地址；行为与旧版一致 |
| A6 | 选中图表/图片后点 Run Analysis | 按 T1.2 谓词视为无选区进入补选流程，不崩溃 |
| A7 | Quick Run / ELISA 无选区 | 最小提示 / 标准品提示按 D1/D2 生效 |

**B. WPS**（长轮询主模式 + 降级开关两模式各验）：

| # | 步骤 | 预期 |
| --- | --- | --- |
| B1 | 不选区点 Run（主模式） | 弹窗置灰 +「选择数据」；点后 WPS 内可框选（promptRange 弹出）；提交后弹窗解锁刷新 |
| B2 | 补选后 OK | 写回 plan 正常执行，图表插入 |
| B3 | 补选等待中点 Ribbon 其他按钮 / cancel | job 被取消，弹窗关闭或回到无数据态，服务不残留 pending |
| B4 | 交互等待超时（>120s 不操作） | 弹窗提示超时可重试；job 未超 300s 总限 |
| B5 | 降级开关启用：不选区点 Run | 先弹 promptRange 补选，再开弹窗且已有数据（无「选择数据」必要） |
| B6 | Quick Run 无选区 | promptRange 预补选后直接出结果 |
| B7 | ELISA 无选区 | 现有双 promptRange 流程正常（回归） |
| B8 | 服务为旧版本（交互端点 404） | JS 自动降级到预补选，无报错 |

---

## 7. 范围、风险与 Git 策略

### 7.1 文件级范围（白名单；对应旧计划 §9.1 语义，供 T6.3 检查）

| 文件 | 动作 |
| --- | --- |
| `xstars/ui_dialog.py` | 修改（M2：构造接口、banner、置灰、刷新、可测试性抽取） |
| `xstars/main.py` | 修改（M3/M4：流程改序、`_prompt_range_windows/_macos`、谓词重构、quick/ELISA 入口） |
| `xstars/application/worker.py` | 修改（M5：无 selection 支持、交互 provider、数据流接通） |
| `xstars/application/contracts.py` | 修改（M5：如需交互 DTO/字段；保持仅 JSON 标量） |
| `xstars/wps_service.py` | 修改（M5：runner 监视、/interaction 端点、OPTIONS 放行） |
| `wps-addon/main.js` | 修改（M5） |
| `wps-addon/service-client.js` | 修改（M5） |
| `wps-addon/config.template.js` | 修改（M5：降级开关字段，载体待定，见待决事项 3） |
| `wps-addon/spreadsheet.js` | 明确不改（promptRange 原样复用） |
| `tests/test_ui_dialog_selection.py` | 新建（M2） |
| `tests/test_macos_support.py`、`tests/test_excel_characterization.py` | 修改（M3/M4 新用例；既有断言不动） |
| `tests/test_wps_worker.py`、`tests/test_wps_service.py` | 修改（M5 新用例） |
| `wps-addon/tests/main.test.cjs`、`wps-addon/tests/service-client.test.cjs` | 修改（M5 新用例） |
| `poc/windows-selection-spike/` | 可选新建（M1 spike 脚本；若提交则仅含 spike 代码与 README） |
| `plans/20261011-run-analysis-in-dialog-selection.md` | 修改（本文件；M1/M7 回填） |
| `ribbon/*`（.bas、customUI14.xml）、`installer/**`、`xstars/tools/**`、`xstars/presets/**`、`docs/**`、分析/统计/绘图引擎（stats_engine.py、plot_engine.py、application/analysis.py 等） | **明确不改** |

### 7.2 风险

| ID | 等级 | 触发条件 | 缓解 |
| --- | --- | --- | --- |
| R-1 | 高 | Tk grab 与 Excel InputBox 焦点死锁（T1.1 证实） | R7 双降级路径已备（grab_release / A1）；spike 先行，B1 不可行即 B2，均不阻塞交付 |
| R-2 | 高 | WPS 内嵌浏览器并发长轮询不可行（T1.3） | D4 降级开关（preprompt）已是设计内备选；S3 失败即切主方案，须经 Frank 确认 |
| R-3 | 中 | ttkbootstrap 主题下 Notebook tab disabled 无视觉灰 | T2.3 备选：单一「请先选数据」占位帧；视觉项进 Mac 手动单 |
| R-4 | 中 | worker 300s 总超时 vs 用户补选耗时（含两次以上重试） | 单次交互等待 120s 有界 + 状态栏提示；超时回无数据态可重发；不改总超时（改 `WORKER_TIMEOUT_SECONDS` 影响面大，列待决事项 4） |
| R-5 | 中 | Windows 补选到其他 sheet / 多区域导致读数或锚点异常 | T3.1 helper 内单区域+同 sheet 校验并友好报错 |
| R-6 | 中 | 跨线程 Tk 更新崩溃（worker 交互 provider） | 队列 + `root.after`/`wait_variable` 主线程 marshal（R13），单测覆盖 |
| R-7 | 中 | 写回锚点语义变化（不再重读当前选区）造成既有用户习惯差异 | golden 测试保证同选区下行为一致；§6.3-2 / §6.4-A3 显式验证锚点 |
| R-8 | 低 | 新旧组件混布（旧 JS + 新服务 / 反之） | 降级路径以 404/INVALID_SELECTION 兜底（T5.5），B8 用例覆盖 |
| R-9 | 低 | 基线失败集记录缺失导致回归误判 | T6.1 强制先记基线快照再对比 |

### 7.3 回滚步骤

1. 功能整体异常：`git revert` 该 PR 的 merge commit（单 PR 策略，一次回滚）；`~/.xstars/wps_service.json` 与加载项文件随安装器分发，回滚后旧版安装包重装即可恢复旧协议（旧 JS 对旧服务零依赖新端点）。
2. 仅 WPS 交互通道异常：加载项侧关闭降级开关（config 字段）即回到 preprompt 模式，无需回滚 Python 侧（旧路径保留为降级路径）。
3. 仅 Windows 补选交互异常：spike 结论若为 B2，InputBox 分支可按 helper 内单点开关退回 A1。
4. 本 Plan 文件回滚即删除（纯文档，无运行时影响）。

### 7.4 待决事项（不拍板，交 Frank / 实施 Agent）

1. **M1 结论落地形态**：B1 vs B2、WPS 主/降级 —— 依 T1.1–T1.3 结果由 Frank 确认（R7/D4 已定「spike 决定」）。
2. **WPS 降级开关默认值**：主模式默认开，还是首版默认 preprompt、验证后再翻转？
3. **降级开关配置载体**：`config.template.js` 注入字段（安装器同步）vs 加载项常量。倾向前者（每实例可配），实施时确认 `scripts/inject-config.cjs` 是否需同步改（若改，须把 `wps-addon/scripts/inject-config.cjs` 补入 §7.1 白名单）。
4. **交互等待时长**：单次 120s / worker 总 300s 是否接受（含「超时后弹窗自动取消」的体验）？
5. **基线失败用例 ID**（证据缺口 5）：实施第一步在 main 上跑全量记录。
6. **Mac 嵌套 simpledialog 在 grab 下的表现**（T4.2）：Mac 本机验证，异常再定独立 Toplevel 方案。

### 7.5 Git 策略

- 分支：`feat/run-dialog-selection`（自 `main` @ `94d938e`）。
- Draft PR：单 PR「feat: allow in-dialog data selection for Run Analysis (Excel Win/Mac + WPS)」，从分支起点即建 Draft，Milestone 证据持续写入。
- 拆分决策：**不拆**。7 个 Milestone 全部服务同一条用户链路（同一 UI 状态机 + 三宿主接线），无多人并行需求，按仓库既有惯例（macos/wps 支持均为单 PR 串行 Milestone）走一分支一 Draft PR；不使用 integration 分支。
- 合入顺序：M1→M7 串行推进，每 Milestone 验证闸门（§5 表）通过才 commit/push 并进入下一 Milestone（沿用 brief/仓库 D-B 惯例）；M7 真机清单全绿后由 Frank 决定转 Ready 与 squash merge。
- PR 描述正文由实施阶段生成，不在本 Plan 写。

---

## Changelog

| rev | 日期 | 变更摘要 | 依据 |
| --- | --- | --- | --- |
| rev 1 | 2026-10-11 | 初版：7 Milestone、9 缺口、三宿主方案（Windows spike 双备选 / Mac A1 / WPS 长轮询+降级）、brief 行号核实（含 main.py:316 机制修正）、访谈决策 D1–D4 固化为 R5–R8 | brief 2026-10-11 + 2026-10-11 访谈（ELISA=A、Quick=B、Windows=C、WPS=A） |


## 实施记录（2026-10-11，rev 1 已批准）

- 本地分支 `feat/run-dialog-selection`，起点 `94d938e`；按本次指令不 commit/push/建 PR。
- M1：T1.1–T1.3 Windows/WPS 无真机，**待 spike 决定**。实现 B1 grab_release/InputBox 与 B2 A1 双路径，不认定任何矩阵格已验证；WPS 长轮询与 preprompt 均保留。
- §7.4-2 Frank 实施期间批准：字段可 true/false，未配置暂走 preprompt，最终生产默认**待 spike 决定**。
- §7.4-3：模板字段可直接注入固定布尔值，无新增占位符，不需改 inject-config.cjs；最终载体/安装器同步不拍板。
- §7.4-4：遵循计划 120s / 300s，不改总限，最终体验待确认。
- T6.1 起点实际基线：`python -m pytest -q --ignore=tests/test_wps_probe.py` → **408 passed, 3 skipped, 24 warnings**（14.80s）；失败集为空，与 brief 的「1 failed」不符，未修任何先存缺陷。
- M2 自动化：UI 新四类状态测试 + Mac/Excel 回归 **75 passed, 2 skipped**；compileall 通过。视觉/嵌套 grab 尚待真机。


### 已批准偏差与边界记录

1. **rev 1 偏差：经批准扩展 WPS 长轮询范围以兑现 R1 重选需求**。配置为 true 时，对全部弹窗类命令（已有选区也包含）并发轮询，而非仅缺失 selection；false/未配置不轮询。
2. 经 supervisor 批准增加请求能力字段 `interactionEnabled: bool`（仅弹窗类），broker 白名单/类型校验，默认 false；worker 在已有选区且能力关闭时隐藏补选按钮，避免 preprompt 模式点击重选后无人响应。缺失 selection 的新服务请求仍可打开空弹窗。旧 broker 也可能因新字段返回 INVALID_REQUEST，客户端与 404/INVALID_SELECTION 一样兼容重试；404 时先取消旧 job，再降级重发。
3. 经 supervisor 批准，单行选区或全空（None/空白字符串）单元格矩阵视为无数据选区。Excel Run/presets/Quick/ELISA 及 WPS Run/presets/Quick 使用同一边界；**有表头但数据为空的多行选区仍交给既有 validate**，不改变有效多行数据处理。Transform-only/Standard Curve 保持原有入口判定，不改；因此保留 `_entry_selection_or_error` 为这些不在范围的入口服务，新增 `_read_valid_selection` 仅接本次入口。
4. 既有两条「Quick 无选区即报错」测试需随已批准 D2 更新：Mac 测试改为最小补选取消、Transform 原错误断言保留；node 原无效选区错误测试改测未纳入范围的 Transform，同断言保留。未修先存缺陷。
5. §6.3-2 示例 `Data!A1:D7` 与 §3.4-C/T4.1 的 active-sheet-only A1 helper 语法不一致；本实现/真实 smoke 用活动表 `A1:B4`，按已有契约拒绝 sheet-qualified、跨表及命名区域；范围横幅仍显示 `sheet!address`。未扩展地址语法。
6. 通用 A1 helper 实测嵌套 simpledialog 可打开且读数成功，但返回时父 grab 被释放；在 finally 显式恢复 `grab_set`/lift，真实 Tk+Excel 重跑通过。不需改独立 Toplevel。

### Milestone 验证与最终证据

- **M1**：Windows S1/S2、WPS S3 均 **待 spike 决定**。Windows `_WINDOWS_SELECTION_MODE` 单点提供 `inputbox`（B1，临时 grab_release/grab_set）与 `a1`（B2，父 Tk A1 输入）。当前 inputbox 分支仅为代码候选，不代表 Frank 已选型；四格矩阵、None/Shape/多区域真实返回、WPS 并发浏览器均无真机结论。
- **M2**：新 `test_ui_dialog_selection.py` 无 Tk 四类状态测试通过；实际 ttkbootstrap 四 tab/OK disabled→normal 在 Mac smoke 中逐项读 widget state，横幅含 n/范围，下拉刷新成功。连续重选幂等与取消、非法友好提示有单测。
- **M3/M4**：Windows/Mac 双平台 mock 补选成功、取消、非法/多区域/跨表重试、范围显示、补选锚点（含再次点击别处）、WB/qPCR labeled 分支、Quick 最小补选、ELISA 标准品补选进入现有拟合 dialog 均覆盖。既有 Mac 样本 A1 与 Windows ELISA InputBox 契约断言保持。
- **M5**：broker GET/POST 鉴权/Origin/字段/200×200 校验、stale ID/取消、并发命令+GET、原子文件握手、worker timeout/cancel 和 payload 数据流；新 E2E 测试用**真实 worker 子进程 + 真实分析/图像 + 假 GUI/JS**，无 selection→GET→POST→writebackPlan（A7/A10）通过，结束无 pending/交互文件残留。node 主/降级、旧服务兼容、已有选区重选、abort、Quick/ELISA 回归均通过。
- **目标行为命令**：`python -m pytest tests/test_ui_dialog_selection.py tests/test_macos_support.py tests/test_excel_characterization.py tests/test_wps_worker.py tests/test_wps_service.py -q` → **151 passed, 2 skipped**（13.96s）。
- **全量最终命令**：`python -m pytest -q --ignore=tests/test_wps_probe.py` → **443 passed, 3 skipped, 27 warnings**（17.99s）；基线 **408 passed, 3 skipped, 24 warnings**；新增 **35 passed**，失败集差集为空。`test_wps_probe` 仍隔离。额外 warnings 为更多图像测试触发已有 matplotlib open-figure 提示，未改绘图算法。
- **加载项**：`cd wps-addon && npm test && npm run check` → **43 passed, 0 failed**，4 文件 node --check 通过；实施前 node 原有 29 测试均仍通过，新增 14。
- **静态**：`python -m compileall -q xstars tests`、`git diff --check` 均退出 0。无 ruff 配置，不运行无配置的 ruff。
- **白名单逐项**：工作树（含未跟踪文件）15 个文件全部属于 §7.1：共享 UI 1；Excel main 1；worker/broker 2；JS/config 3；Python tests 5；node tests 2；本 Plan 1。`contracts.py` 无需新增 DTO，不改；spike 脚本未提交；inject-config.cjs 不改。名单检查用 `git diff --name-only 94d938e` + `git ls-files --others --exclude-standard`，不是仅依赖空提交 diff。
- **保护范围**：`git diff --exit-code 94d938e -- ribbon 'xstars/tools/**' 'xstars/presets/**' installer docs xstars/stats_engine.py xstars/plot_engine.py xstars/application/analysis.py wps-addon/spreadsheet.js` 零 diff；AST 比对 `_run_transform_only_impl`、`_run_standard_curve_impl`、`_select_sample_data`、`TransformOnlyDialog` 完全相同；WPS `runElisa` 源码完全相同。
- **Git**：§6.1 原命令 `git diff --name-only origin/main...HEAD` 与 VBA 零 diff 亦运行通过，但因为本次明确禁止 commit，HEAD 未变，实际验收以以上工作树检查为准。`git diff --cached --name-only` 为空；无 commit/push/PR，仍 HEAD `94d938e`。

### Mac 真机 smoke（不等同完整安装版 Ribbon 验收）

- 已按 Frank 指示先自动拉起 Excel：`osascript ... Microsoft Excel ... version` → **16.112.3**，成功。
- 本地一次性脚本 `/tmp/xstars-mac-selection-smoke.py`（不入仓库）：xlwings 新临时工作簿/活动空白 D20 → 调真实 `_run_impl(book)` → 真实 SettingsDialog 四 tab/OK disabled → 自动填写真实嵌套 simpledialog A1:B4 → n/范围/Control 下拉+grab 恢复 → OK → 真实统计写入 A7、**1 张分析图**。输出：`REAL_TK_A1_GRAB_AND_REFRESH_OK`、统计值 Independent t-test / p=0.0002552167494419273。临时未保存簿触发现有 artifact warning（提示先保存才能 export），不影响分析写回；未修此既有行为。
- 已选 A1:B4 的真实 `book.selection` 经 `_read_valid_selection` 返回同地址，验证 Mac 有效选区谓词不是误判。
- `/tmp/xstars-wps-tk-cancel-smoke.py` 用真实 Tk 取消正在等文件交互的 provider，输出 `REAL_TK_WORKER_CANCEL_DURING_FILE_WAIT_OK True`，无异常弹栈/残留 Tk 回调。
- 初期 UI 自动填充选错 Entry 的 smoke 两次超时（不作为产品通过证据）；改为定向 QueryString.entry 后定位父 grab 问题并修复重跑。所有创建的临时工作簿按名称及 SelectionSmoke 表名核对后关闭，未关闭原 XSTARS_Templates.xlsx。
- 未执行完整 §6.3 的安装版 Ribbon 入口、取消/非法地址视觉、Quick/ELISA/presets 全链路手工清单；因此 **T4.4/T6.4 保留未勾选，M4/M6 标记 [~]（代码/自动化通过，手工项未齐）**。

### 未完成与 §7.4 建议（非拍板）

- M1 S1–S3 与 M7 全部 **待 spike 决定 / Frank 真机验证**；Mac §6.3 剩余手工项同上。
- §7.4-1：建议先实测 B1（释放 grab）可用则保留框选，不可用切 B2 A1；WPS 并发不可用则主用 preprompt，均由 Frank 决定。
- §7.4-2：建议保持已批准临时未配置/false→preprompt，Windows/WPS 真机通过后再决定是否 true 为生产默认。
- §7.4-3：建议 per-instance `config.template.js` 的 `inDialogSelection` 字段，当前无需注入脚本改动；安装器覆盖/保留用户值的策略仍由 Frank 决定，本次不改 installer。
- §7.4-4：单次120s / 总300s 原约束不变；timeout/取消文案明确可重试，不私自加长总限。
- §7.4-5：基线事实已消除证据缺口，失败集为空。
- §7.4-6：真实嵌套 simpledialog+grab 恢复已通过；建议沿用父 Tk 会话，不引独立 Toplevel；完整安装版手工验收仍待补证。

### Code review 修复复验（本地工作树，仍不 commit/push）

- MAJOR-1：`/interaction` 选区改按 `SelectionPayload.from_dict` + values/address/sheet 存在性校验，允许与 `/command` 相同的 workbook 元数据。node fake sheet 增加真实 Parent.Name，交互 POST 断言 workbook；broker 实际 HTTP POST 带 workbook 成功并可被 worker 消费。
- MAJOR-2：`_show_error(parent_root=...)` 复用已有 root，不创建/销毁第二 root；provider、Windows 范围错误与 Mac helper 失败均传父 root。新增 messagebox parent 与 provider 重试断言。
- MINOR-1：先检查行数/首单元格；非空首格不再读取整矩阵，首格空时仍保留全矩阵判断以免拒绝后续有数据的选区。Mac 弹窗 provider/Quick 的已读 DataFrame 传给 `_read_selection_auto` 复用，ELISA 标准品也复用已读 frame；不引跨请求缓存，不改数据清洗/分析算法。
- MINOR-2/3：补选回填 expectedContext（含 workbook）；任何 poll/picker 通道异常均取消服务 job 并 abort 请求。新增预补选/交互模式 workbook 漂移守卫、GET 网络/JSON、POST 服务错误/JSON、picker 异常取消测试。
- MINOR-4：保留仍有效的 control/blank/comparison 选择；重选不再强制清除 reference 开关，首次无数据→带标签列仍设置默认 True。
- MINOR-5：新增**真实 Tk event loop + 真实文件 I/O + 真实线程**回归，不 mock `_wait_dialog_interaction`、after、wait_variable 或 BooleanVar；覆盖成功、外部取消、root 销毁、非法 response 和已有选区重新补选。根窗口仅主线程操作；取消/异常均清理 after 回调与 Destroy binding。
- MINOR-6：无选区且 interactionEnabled=false（含缺省 false）在 broker 入 job 前返回 400，避免无人轮询的空弹窗。
- NIT-1/2/3/4/5/6：deadline 后查 response；两新交互码加入 USER_MESSAGES；provider Exception 统一父 messagebox；paired 重新 pack 在 Show ns 前；Windows/Mac/JS prompt 均拒绝单行/全空；有效选区谓词仅吞已知非 Range 类型异常，其他宿主故障记日志并向上传播。
- 新测试首轮曾有两个 PropertyMock 未指定 create=True 的断言准备错误，已修复；真实 Tk 测试在同进程后续 broker 线程触发 macOS Trace/BPT trap，判断为测试回调循环对象由后续线程 GC 回收，fixture 收尾增加主线程 gc.collect 后，目标与全量命令均正常完成。此为测试隔离收尾，不改变生产 Tk 架构。
- 最终目标测试：**172 passed, 2 skipped**（原 151 passed, 2 skipped）。全量 `python -m pytest -q --ignore=tests/test_wps_probe.py`：**464 passed, 3 skipped, 27 warnings**（原 443 passed, 3 skipped, 27 warnings），新增 21 passed、无新增失败；真实 Tk 新用例本机均执行通过、未跳过。新增 Quick frame 复用测试显式关闭自己生成的 figure，以免扰动后续测试已有 open-figure warning 阈值。
- node `npm test && npm run check`：**55 passed, 0 failed**（原 43 passed，新增 12）；compileall、git diff --check 通过。
- 白名单：含未跟踪文件的 15 个工作树路径仍全部属于 §7.1；保护范围（ribbon/installer/tools/presets/docs/算法/contracts/spreadsheet.js）零 diff；受保护 Python AST 与 WPS runElisa 源码比对不变。cached diff 为空；HEAD 仍 `94d938e`、分支仍 `feat/run-dialog-selection`。
- Windows/WPS 真机与完整安装版 Mac Ribbon 清单仍待原责任人执行；生产开关仍 false，不擅自改变 M1/M7 决策。
