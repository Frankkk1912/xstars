# XSTARS WPS Gate 0 PoC

本目录用于执行 `docs/wps-support-implementation-plan.md` 的 M0 可行性硬门槛。它不是正式产品代码；Gate 0 通过前不得据此开始大规模核心重构。

## 当前状态

- M0.1：**企业授权实机阻断验收已通过**。
- M0.2：**实机写回/图片持久化验收已通过**（2026-08-29，详见下文记录）。
- M0.3：**实机验证完成，含关键负面发现：`OAAssist.ShellExecute` 在本宿主失效，拉起策略改为安装器自启动**（详见下文记录）。
- 测试宿主：WPS 365 教育高级版 `12.1.0.28022` 64 位，完全断网。
- 官方 `publish` 安装、Ribbon、按钮回调、完全退出后重启回调和卸载均通过；实际 Origin 为 `file://`。
- Node.js 与 npm 可用；PoC 固定使用官方 `wpsjs 2.2.3`。
- 已用 `wpsjs create xstars-wps-poc` 生成“电子表格 + 无 UI 框架”模板，再缩减为单个 Ribbon 回调按钮。

## 官方依据

- [WPS 加载项开发说明](https://open.wps.cn/documents/app-integration-dev/wps365/client/wpsoffice/wps-integration-mode/wps-addin-development/wps-addin-development-instructions)
- [生成首个 WPS 加载项](https://open.wps.cn/documents/app-integration-dev/wps365/client/wpsoffice/wps-integration-mode/wps-addin-development/generate-the-first-wps-addin)
- [WPS 加载项可用性](https://open.wps.cn/documents/app-integration-dev/wps365/client/wpsoffice/jsapi/addin-api/wps-addin-availability)

官方当前推荐 `publish` 模式；`jsplugins.xml` 模式从 WPS `12.1.0.16910` 起受到限制，因此本 PoC 不把手写 `oem.ini/jsplugins.xml` 当作正式部署结论。

## 复现开发构建

```powershell
cd poc/wps/addin
npm.cmd ci
npm.cmd run build
```

如果 PowerShell 禁止运行 `npm.ps1`，应直接使用 `npm.cmd`，不需要放宽系统执行策略。

开发调试命令为 `npm run wps:debug`。该命令会启动 WPS 和本地开发服务，只用于联机调试，不等于完全断网的 Gate 0 部署验收。

## 生成官方离线发布物

在 `poc/wps/addin` 中执行：

```powershell
npm.cmd run wps:package:offline
python -m http.server 3890 --bind 127.0.0.1 --directory deploy
```

`wps:package:offline` 调用官方 `wpsjs` 的离线构建与 `publish` 流程，并将可部署目录暂存到 `deploy/`。浏览器安装入口为 `http://127.0.0.1:3890/publish.html`。

本次实测生成：

- `wps-addon-build/xstars-wps-poc.7z`；
- `wps-addon-publish/publish.html`；
- 发布记录：`addonType=et`、`online=false`、`multiUser=false`、`version=1.1.0`；
- 预期插件 URL：`http://127.0.0.1:3890/addin/xstars-wps-poc.7z`。

### `wpsjs 2.2.3` 发布注意事项

`wpsjs 2.2.3` 包内的 `publishlist.json` 可能携带示例记录，并会被 `publish` 命令原地更新。本次生成页面曾出现与 XSTARS 无关的 `test123` 记录。因此：

1. 不提交生成的 `publish.html` 或 `.7z`；
2. `scripts/build-publish-offline.cjs` 会临时隔离包内列表、调用官方发布器、恢复包文件，并断言生成页面只包含 `xstars-wps-poc`；
3. 实机安装前仍须复核 `publish.html` 的嵌入列表和实际 Origin；
4. `npm audit` 的剩余高危项来自 `wpsjs → inquirer → tmp`，上游暂无修复；该工具只能在可信仓库、可信输入下作为开发依赖运行，不能进入产品运行时。

## M0.1 实机验收记录

必须在真实 WPS 专业/商业版 x64、完全断网环境完成：

- [x] 记录 WPS 完整版本和版本类型；
- [x] 通过官方 `publish` 流程安装/启用离线加载项；
- [x] 显示 `XSTARS Gate 0` Ribbon；
- [x] 点击“验证加载项回调”并看到当前工作簿名；
- [x] 记录加载项实际 Origin：`file://`；
- [x] 关闭网络后重复启动、回调和卸载流程；
- [ ] 在个人版执行相同 smoke test，仅记录能力，不作为阻断结果。

企业授权阻断项已全部通过；个人版 Beta smoke test 尚未执行，但不阻断 M0.2。

## M0.2 Selection/Value2/AddPicture 实机验证

M0.2 使用独立端口：`3890` 仅提供官方安装页，`3891` 仅提供回环 JSON 探针。探针只绑定 `127.0.0.1`，并针对已观测的 `file://` 页面严格允许 CORS Origin `null`，不使用通配符。

保持完全断网，打开两个 PowerShell 窗口。

窗口 A（仓库根目录）：

```powershell
cd E:\Documents\GitHub\xstars
python poc\wps\probe_server.py --port 3891
```

窗口 B（加载项目录）：

```powershell
cd E:\Documents\GitHub\xstars\poc\wps\addin
python -m http.server 3890 --bind 127.0.0.1 --directory deploy
```

打开 `http://127.0.0.1:3890/publish.html`，安装 `xstars-wps-poc 1.1.0`。随后：

1. 在新工作表输入一个小型二维矩阵，例如 `A1:B2 = [[1, 2], [3, 4]]`；
2. 选择 `A1:B2`，点击 `XSTARS Gate 0 → 验证选区写回/图片`；
3. 验证探针收到 `OPTIONS` 和 `POST /probe`，WPS 在 `D1:E2` 写回同一矩阵，并在其下方插入 PNG；
4. 将工作簿另存为测试副本，完全退出并重开 WPS；
5. 确认写回值和图片仍存在，记录图片位置、尺寸及任何错误。

- [x] `Selection.Address()` 和二维 `Value2` 读取正确；
- [x] Origin `null` 的预检和 JSON 往返成功；
- [x] 二维 `Value2` 批量写回正确；
- [x] `Shapes.AddPicture(path, 0, -1, ...)` 成功嵌入 PNG；
- [x] 保存并重开后写回值和图片仍存在。

## M0.3 实机验收记录（2026-08-29）

宿主同前：WPS 365 教育高级版 `12.1.0.28022` 64 位，完全断网。

### 关键发现：`OAAssist.ShellExecute` 失效

- 实机点击「拉起本地服务」：弹出官方安全确认窗，点击「是」后 ShellExecute 返回 `null`，**目标进程从未被创建**。
- 铁证：服务脚本入口首行写日志（`process entry argv=...`）+ 5 分钟 143 次进程/端口采样，日志始终空白、pythonw 进程从未出现；而同一命令行经 PowerShell 直接拉起 pythonw 立即成功。
- 官方签名实为 2 参数 `ShellExecute(Url, Params)`（传 Windows API 5 参数版会被 `too many parameters` 拒绝）；修正后依然弹窗后无动作，与社区报告的接口审查下线行为一致。
- **策略结论：加载项不能自行拉起本地服务；服务自启动由 M5.1 独立安装器负责，加载项仅做健康检查与用户引导**。该发现是 Gate 0 的核心产出之一。

### 其余验证项（服务以安装器等价方式预先启动）

- [x] 幂等路径：服务已运行时 `/health` 探测成功并提示「已在运行」，不重复拉起（单元测试覆盖；冲突诊断按钮的存活检查亦实时验证了 WPS→服务 JSON 往返）；
- [x] 真实 Origin 与预检：WPS 端 fetch 实际携带 `Origin: file://`，`OPTIONS` 预检与 `/dialog` JSON 往返成功；
- [x] Tkinter 阻塞/取消：模态对话框打开期间服务探活正常、WPS 不冻结（实机确认）；「确定」与「取消」均被正确回报（日志 `confirmed=True/False`）；
- [x] 端口冲突可诊断：服务端自测通过；真实双开产生 `PORT CONFLICT` 日志（WinError 10048）且第二实例以退出码 2 退出，原服务不受影响（PID 不变）；实现采用 Windows `SO_EXCLUSIVEADDRUSE`（`SO_REUSEADDR` 在 Windows 上会静默双开，自测已验证该陷阱）；
- [x] 可恢复性：服务被杀后可由外部重新拉起并立即恢复服务；加载项内拉起不可用（见上）。

- 诊断链路：`/diagnostics` 返回日志尾部；日志位于 `poc/wps/.gate0-artifacts/service.log`；`/health` 上报真实 os_pid。

## M0.2 实机验收记录（2026-08-29）

完全断网，宿主为 WPS 365 教育高级版 `12.1.0.28022` 64 位。

- 官方 `publish` 安装 `xstars-wps-poc 1.1.0` 成功（3890 日志记录 `addin/xstars-wps-poc.7z` 多次 200 下载）。
- 选中 `A1:B2 = [[1, 2], [3, 4]]` 后点击“验证选区写回/图片”，探针在 17:00:33 生成 PNG 工件，证明 `OPTIONS` + `POST /probe` JSON 往返成功。
- `D1:E2` 写回矩阵与 `A1:B2` 完全一致。
- PNG 嵌入在 `D1:E2` 下方，尺寸 11.29 × 6.35 厘米（宽 × 高，320×180 像素 @72 DPI 原生尺寸）。
- 另存副本、完全退出并重开 WPS 后，写回值和图片均仍存在。
- 全程无任何报错或异常提示。

## M0.3 服务拉起/CORS/Tkinter/生命周期实机验证

M0.3 使用独立端口：`3890` 提供官方安装页（升级到 `1.2.0`），`3892` 是被验证的本地服务。
**不要预先启动 3892**——服务拉起本身就是要验证的 `OAAssist.ShellExecute` 流程。加载项会通过 `pythonw` 无窗口拉起 `poc/wps/service_server.py`（路径硬编码在 `js/ribbon.js` 的 `XSTARS_GATE0_SERVICE`，仅限本机 PoC）。

服务端自测（已通过，可随时复跑）：

```powershell
cd E:\Documents\GitHub\xstars
python poc\wps\service_server.py --self-test
```

实机步骤（保持完全断网）：

1. 打开 `http://127.0.0.1:3890/publish.html`，将 `xstars-wps-poc` 从 1.1.0 升级安装到 1.2.0；
2. 点击 `XSTARS Gate 0 → 拉起本地服务`：预期弹出「服务拉起成功」，含 PID 和服务端记录 Origin；此时不应有任何控制台窗口（pythonw 无窗口拉起）；
3. 再次点击「拉起本地服务」：预期提示「已在运行」（幂等路径）；
4. 点击「测试 Tkinter 对话框」：预期弹出 Tkinter 模态对话框；**对话框打开期间在 WPS 里点击单元格、切换 Sheet，确认 WPS 不冻结**；然后点击「取消」；预期 alert 显示「选择：取消」和三次健康探活延迟（非「无响应」）；
5. 点击「端口冲突诊断」：预期原服务 PID 不变，并展示第二实例的 `PORT CONFLICT` 日志记录；
6. 可恢复性：运行 `taskkill /IM pythonw.exe /F` 强杀服务后，再点「拉起本地服务」，预期重新拉起成功。

诊断日志位于 `poc/wps/.gate0-artifacts/service.log`；服务端关键实现为 Windows `SO_EXCLUSIVEADDRUSE` 单实例绑定（`SO_REUSEADDR` 在 Windows 上会静默双开，自测已验证该陷阱）。

- [ ] `OAAssist.ShellExecute` 能无窗口拉起本地服务并就绪；
- [ ] 服务已在运行时幂等报告，不重复拉起；
- [ ] 真实 Origin（`null`）预检和 JSON 往返成功；
- [ ] Tkinter 对话框打开期间服务探活成功、WPS 不冻结，取消可被正确回报；
- [ ] 第二实例端口冲突失败有诊断日志，原服务不受影响；
- [ ] 服务被强杀后可通过 Ribbon 重新拉起。

## M0.4 实机验证清单（待执行）

责任人：用户。宿主要求：真实 WPS 专业版/商业版/政企版 x64，完全断网。每步填写版本、结果、截图/文件路径或哈希及异常信息；M0.4 全部阻断项通过后才可进入下一 Milestone。

### 0. 环境准备

1. 在仓库根目录启动 3892 服务：

   ```powershell
   cd E:\Documents\GitHub\xstars
   .\.venv\Scripts\python.exe poc\wps\service_server.py --port 3892
   ```

2. 在 `poc/wps/addin` 重新生成官方离线包：`npm.cmd ci`，再运行 `npm.cmd run wps:package:offline`；不要手写 `publish.xml`。
3. 启动离线安装页：`python -m http.server 3890 --bind 127.0.0.1 --directory deploy`，在 WPS 中升级安装当前 `xstars-wps-poc`，确认出现「M0.4 探针」组。
4. 记录：Windows / WPS 完整版本、授权类型、Python / Pillow 版本、显示器缩放比例、加载项实际 Origin。

- 结果/备注：

### 1. InputBox(Type=8) 选区与取消

1. 准备一个小型 ELISA 风格二维矩阵；点击「InputBox 选区探针」，用鼠标框选区域。
2. 确认返回地址、行列数和非空单元格数正确。
3. 再次点击并取消 InputBox，确认提示「用户取消（非错误）」且服务没有收到错误请求。

- 结果/备注：

### 2. 两阶段三击流程

1. 框选标准品区域，第一次点击「两阶段选区探针」，确认回显标准品地址。
2. 框选样本区域，第二次点击，确认回显样本地址。
3. 第三次点击提交，确认 3892 服务收到两个矩形二维选区；再次点击应从「标准品」阶段重新开始。

- 结果/备注：

### 3. 地址输入兜底

代码入口 `runM04AddressFallback()` 已存在，本轮只确认代码可被后续产品路径调用；不重复执行 Tkinter/地址交互实机验证。

- 结果/备注：

### 4. 选中图片时的 Selection 对象

选中一张 WPS 图片/Shape，点击「Shape 导出探针」，记录 alert 中的 `Selection.Type`、是否暴露 `ShapeRange`、Shape 名称（如可见）以及主调用是否使用 `CopyPicture(2, -4147)`。

- 结果/备注：

### 5. CopyPicture → 剪贴板 → 四格式/三 DPI 导出

对同一张含文字和细线的图片逐项执行下表。每次点击「Shape 导出探针」，输入格式与 DPI；记录输出路径、像素尺寸、DPI 元数据、文件哈希及能否由独立查看器打开。

| 格式 | 96 DPI | 300 DPI | 600 DPI |
| --- | --- | --- | --- |
| PNG | 结果/备注： | 结果/备注： | 结果/备注： |
| TIFF | 结果/备注： | 结果/备注： | 结果/备注： |
| JPG | 结果/备注： | 结果/备注： | 结果/备注： |
| PDF | 结果/备注： | 结果/备注： | 结果/备注： |

### 6. 125% / 150% 显示缩放对比

分别在 125% 与 150% Windows 显示缩放下重复至少 PNG 300/600 DPI；核对像素尺寸、DPI 元数据和视觉清晰度是否随显示缩放产生非预期变化。

- 125% 结果/备注：
- 150% 结果/备注：

### 7. 保存并重开

保存并完全退出 WPS，重开工作簿，确认源图片/Shape 仍存在且可再次导出；确认已生成的四格式文件仍可打开、大小与哈希未变化。

- 结果/备注：

### 8. COM `Ket.Application` 对照探测

「Shape 导出探针」在导出后自动请求 `/probe/com-probe`。记录 `GetActiveObject("Ket.Application")` 成功时的 WPS Version，或失败时的 `COM_UNAVAILABLE` 异常类型/错误码；同时记录 WPS 与 Python 服务的 UAC 权限级别是否一致。

- 结果/备注：

### 9. CF_ENHMETAFILE 矢量质量评估

根据服务错误诊断中的剪贴板格式以及 96/300/600 DPI 输出，记录是否存在增强型图元文件、细线/文字在高 DPI 下是否真正增加细节。若 Pillow 仅得到显示分辨率 DIB 且上采样无新增细节，明确标记为导出等价性阻断证据，不得静默判定通过。

- 结果/备注：

### M0.4 结论（执行后填写）

- ELISA 最终交互路径：
- Shape 导出最终路径（或 O3 fallback 决策需求）：
- 阻断项：
- WPS/Windows/Python/Pillow 版本：
- 截图、导出文件与 SHA-256：
- 执行日期与执行人：
