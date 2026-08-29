# XSTARS WPS Gate 0 PoC

本目录用于执行 `docs/wps-support-implementation-plan.md` 的 M0 可行性硬门槛。它不是正式产品代码；Gate 0 通过前不得据此开始大规模核心重构。

## 当前状态

- M0.1：**企业授权实机阻断验收已通过**。
- M0.2：**实机写回/图片持久化验收已通过**（2026-08-29，详见下文记录）。
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

## M0.2 实机验收记录（2026-08-29）

完全断网，宿主为 WPS 365 教育高级版 `12.1.0.28022` 64 位。

- 官方 `publish` 安装 `xstars-wps-poc 1.1.0` 成功（3890 日志记录 `addin/xstars-wps-poc.7z` 多次 200 下载）。
- 选中 `A1:B2 = [[1, 2], [3, 4]]` 后点击“验证选区写回/图片”，探针在 17:00:33 生成 PNG 工件，证明 `OPTIONS` + `POST /probe` JSON 往返成功。
- `D1:E2` 写回矩阵与 `A1:B2` 完全一致。
- PNG 嵌入在 `D1:E2` 下方，尺寸 11.29 × 6.35 厘米（宽 × 高，320×180 像素 @72 DPI 原生尺寸）。
- 另存副本、完全退出并重开 WPS 后，写回值和图片均仍存在。
- 全程无任何报错或异常提示。
