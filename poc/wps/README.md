# XSTARS WPS Gate 0 PoC

本目录用于执行 `docs/wps-support-implementation-plan.md` 的 M0 可行性硬门槛。它不是正式产品代码；Gate 0 通过前不得据此开始大规模核心重构。

## 当前状态

- M0.1：**准备中，尚未通过实机阻断验收**。
- 本机发现 WPS Office `12.1.0.28022`，但尚未证明其属于阻断目标中的专业版/商业版/政企版。
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
npm ci
npm run build
```

开发调试命令为 `npm run wps:debug`。该命令会启动 WPS 和本地开发服务，只用于联机调试，不等于完全断网的 Gate 0 部署验收。

## 生成官方离线发布物

在 `poc/wps/addin` 中执行：

```powershell
npm run wps:package:offline
python -m http.server 3890 --bind 127.0.0.1 --directory deploy
```

`wps:package:offline` 调用官方 `wpsjs` 的离线构建与 `publish` 流程，并将可部署目录暂存到 `deploy/`。浏览器安装入口为 `http://127.0.0.1:3890/publish.html`。

本次实测生成：

- `wps-addon-build/xstars-wps-poc.7z`；
- `wps-addon-publish/publish.html`；
- 发布记录：`addonType=et`、`online=false`、`multiUser=false`、`version=1.0.0`；
- 预期插件 URL：`http://127.0.0.1:3890/addin/xstars-wps-poc.7z`。

### `wpsjs 2.2.3` 发布注意事项

`wpsjs 2.2.3` 包内的 `publishlist.json` 可能携带示例记录，并会被 `publish` 命令原地更新。本次生成页面曾出现与 XSTARS 无关的 `test123` 记录。因此：

1. 不提交生成的 `publish.html` 或 `.7z`；
2. `scripts/build-publish-offline.cjs` 会临时隔离包内列表、调用官方发布器、恢复包文件，并断言生成页面只包含 `xstars-wps-poc`；
3. 实机安装前仍须复核 `publish.html` 的嵌入列表和实际 Origin；
4. `npm audit` 的剩余高危项来自 `wpsjs → inquirer → tmp`，上游暂无修复；该工具只能在可信仓库、可信输入下作为开发依赖运行，不能进入产品运行时。

## M0.1 实机验收记录

必须在真实 WPS 专业/商业版 x64、完全断网环境完成：

- [ ] 记录 WPS 完整版本和版本类型；
- [ ] 通过官方 `publish` 流程安装/启用离线加载项；
- [ ] 显示 `XSTARS Gate 0` Ribbon；
- [ ] 点击“验证加载项回调”并看到当前工作簿名；
- [ ] 记录加载项实际 Origin；
- [ ] 关闭网络后重复启动、回调和卸载流程；
- [ ] 在个人版执行相同 smoke test，仅记录能力，不作为阻断结果。

任何一项未验证都不视为 M0.1 通过。
