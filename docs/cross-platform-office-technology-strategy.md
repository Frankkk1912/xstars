# XSTARS 跨平台 Office 技术路线

> 状态：已确定为后续架构方向
> 范围：Microsoft Excel / WPS 表格，Windows / macOS

## 结论

XSTARS 不追求用一种宿主插件技术覆盖所有平台，而采用：

> **统一 Python 核心与通信契约，按宿主提供独立适配器。**

近期技术路线：

| 平台 | 推荐技术 | 状态 |
| --- | --- | --- |
| Excel Windows | VBA + xlwings | 保留现有实现 |
| Excel macOS | VBA + xlwings | 后续增加 Mac 适配 |
| WPS Windows | WPS JS 加载项 + 本地 Python 服务 | 当前规划路线 |
| WPS macOS | 独立 PoC 后决定 | 暂不承诺正式支持 |

长期如需统一现代任务窗格 UI，可调整为：

- Excel Windows/macOS：Office.js；
- WPS Windows/macOS：WPS JSAPI（以 WPS macOS 实测支持为前提）；
- 两套 Manifest、两个 Host Adapter，共享 TypeScript UI、RPC 协议和 Python Core。

## 选择依据

### 1. 没有可覆盖四端的单一插件技术

- VBA 可用于 Excel Windows/macOS，但不能作为 WPS 全平台的可靠基础；
- xlwings 官方支持 Microsoft Excel Windows/macOS，但不正式支持 WPS；
- Office.js 支持 Excel Windows/macOS，但不能直接运行于 WPS；
- WPS JSAPI 与 Office.js 使用不同的 Manifest、对象模型和部署方式；
- WPS 官方加载项发布文档目前只明确列出 Windows/Linux 企业版分支，尚不足以证明 WPS macOS 可按相同方式部署。

因此只能统一业务核心，不能统一宿主接入代码。

### 2. Excel macOS 继续使用 xlwings 成本最低

xlwings 官方支持 Excel macOS，其 Add-in 和 `RunPython` 可使用 Mac Python 解释器。macOS 不支持 xlwings UDF，但 XSTARS 当前不依赖 UDF，因此不是主要障碍。

需要单独适配的是安装、路径、AppleScript/App Sandbox、签名公证和图片导出，而不是重写全部 Excel 前端。

### 3. WPS Windows 应避免依赖 VBA

WPS 的 VBA 能力受版本和授权影响，无法稳定覆盖个人版。WPS JS 加载项是更合理的产品化入口：

- Ribbon 与业务回调由 WPS JSAPI 实现；
- 选区读取和结果写回留在 WPS 插件端；
- 统计、预设和绘图继续由本地 Python 完成；
- 完全离线时仅使用 `127.0.0.1` 回环通信。

### 4. 暂不为“技术统一”迁移 Excel 到 Office.js

Office.js 是 Microsoft 的跨平台方案，但它本质上是 Web Add-in，生产部署通常需要 Web 资源、HTTPS、Manifest 和沙箱内通信。对于 XSTARS 的完全离线本地 Python 模式，会额外引入本地 HTTPS、证书、CORS、服务发现和 macOS 签名公证问题。

在现有 Excel VBA/xlwings 已可用的情况下，立即迁移 Office.js 的收益不足以覆盖重写成本和风险。

## 目标架构

```text
                   XSTARS Python Core
       统计 / 预设 / 绘图 / 配置 / 高分辨率导出
                            │
                 Application Contracts
      AnalysisRequest -> WritebackPlan + Artifacts
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
   Direct Python Bridge                 Local RPC Bridge
          │                                   │
  Excel VBA + xlwings                  WPS JS Add-in
  ├─ Windows adapter                   ├─ Windows adapter
  └─ macOS adapter                     └─ macOS adapter（待验证）
```

实施原则：

1. Python Core 不直接依赖 Excel/WPS API；
2. 使用通用名称，如 `AnalysisRequest`、`WritebackPlan`、`HostCapabilities`，不在核心契约中写死 WPS；
3. Excel 与 WPS 可以使用不同传输方式，不强求全部改成 HTTP；
4. 每个宿主独立负责选区读取、结果写回、图片插入和状态提示；
5. Shape 导出、二次选区、状态栏等差异通过能力检测处理，不假设所有宿主功能相同；
6. Windows 与 macOS 使用独立安装包和签名流程；
7. WPS macOS 必须先完成加载项、离线部署、本地服务通信和图片写回 PoC，再决定是否进入正式支持范围。

## 不采用的方案

- **全平台 VBA**：无法可靠覆盖 WPS，尤其是个人版和 macOS；
- **全平台 JS 宏/JSA**：Office 与 WPS 格式/API 不兼容，也不适合大型插件的升级和部署；
- **现在全面迁移 Office.js**：增加离线部署和本地服务复杂度，同时重写已有 Excel 功能；
- **假定 WPS macOS 与 Windows 能力一致**：目前缺少充分官方证据，必须实机验证。

## 后续决策门槛

在正式规划 WPS macOS 前，至少验证：

- WPS JS 加载项是否可安装并长期启用；
- Ribbon、Selection、Value2、Shapes 等 API 是否可用；
- 离线发布配置及沙箱路径；
- 插件能否访问本地已签名/公证的 Python 服务；
- 图片写回与高分辨率导出能力。

若上述能力不足，则 WPS macOS 应采用独立桌面应用/文件交换作为备选，而不是破坏其他平台的稳定架构。

## 参考资料

- xlwings 平台与依赖：<https://docs.xlwings.org/en/stable/installation.html>
- xlwings Add-in 与 macOS：<https://docs.xlwings.org/en/stable/addin.html>
- Microsoft Office Add-ins 概览：<https://learn.microsoft.com/en-us/office/dev/add-ins/overview/office-add-ins>
- Microsoft Office Add-in Manifest：<https://learn.microsoft.com/en-us/office/dev/add-ins/develop/add-in-manifests>
- WPS 加载项开发说明：<https://open.wps.cn/documents/app-integration-dev/wps365/client/wpsoffice/wps-integration-mode/wps-addin-development/wps-addin-development-instructions>
- WPS 加载项可用性：<https://open.wps.cn/documents/app-integration-dev/wps365/client/wpsoffice/jsapi/addin-api/wps-addin-availability>
