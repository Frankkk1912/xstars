# XSTARS macOS .pkg 真机验收 Runbook（逐项打勾用）

> 对应：Plan §8.6（V-01~V-05）+ §8.4（卸载 U-01/U-02），回填目标 = Draft PR #6。
> 制品：CI run 35317716195 的 `macos-installer-pkg`（136MiB，retention 至 2026-09-25）。
> 环境：Mac mini，Apple Silicon，macOS 14+（Sonoma）。
> 记录约定：每项写 Pass / Fail / Deviation（偏离）+ 一行证据（命令输出或截图路径）。

---

## P0 冲突预检（装机后、开 Excel 前做一次）

背景：你之前走 dev/手动回调装过 XSTARS.xlam。pkg 的 postinstall 已直接覆盖 Startup 文件夹里的
XSTARS.xlam（不会双载），共享 xlwings 文件已做安装前备份；**但旧的 OPEN/OPENn 注册值若指向
其它路径的 XSTARS.xlam，Excel 会双载** → 必须先确认。

- [ ] **P0-1 Startup 文件夹里是 pkg 版本**

  ```bash
  ls -l "$HOME/Library/Group Containers/UBF8T346G9.Office/User Content.localized/Startup.localized/Excel/"
  ```

  预期：`XSTARS.xlam` 存在，修改时间 ≈ 你装 pkg 的时刻；同目录若还有 dev 时代手动放的其它
  XSTARS 副本（不同文件名），记下来。

- [ ] **P0-2 preinstall 备份三态已记录**（共享文件回滚的凭证）

  ```bash
  ls -la "$HOME/Library/Application Support/XSTARS/preinstall-backup/"
  ```

  预期：`xlwings.applescript` 与 `xlwings.conf` 各自要么有备份文件、要么有 `.xlwings.applescript.absent`
  / `.xlwings.conf.absent` marker（二选一，两者都要有记录）。若 dev 模式留过旧 conf，这里应有备份。

- [ ] **P0-3 当前 conf 的 INTERPRETER_MAC 指向内置运行时**

  ```bash
  cat "$HOME/Library/Containers/com.microsoft.Excel/Data/xlwings.conf"
  ```

  预期：`"INTERPRETER_MAC","$HOME/Library/Application Support/XSTARS/python/bin/python3"`；
  若你 dev conf 里有其它自定义键，它们应被保留（只换 INTERPRETER_MAC 行）。

- [ ] **P0-4 注册库无指向其它路径的 XSTARS 加载值（防双载）**

  ```bash
  for db in "$HOME/Library/Group Containers/UBF8T346G9.Office/MicrosoftRegistrationDB/"*.reg; do
    [ -f "$db" ] || continue
    echo "== $db"
    sqlite3 "$db" "SELECT name, CAST(value AS TEXT) FROM HKEY_CURRENT_USER_values
      WHERE UPPER(name)='OPEN' OR UPPER(name) GLOB 'OPEN[0-9]*';" 2>/dev/null
  done
  ```

  预期：所有 OPEN* 值要么不含 XSTARS 字样，要么只指向
  `.../Startup.localized/Excel/XSTARS.xlam`。**若出现其它路径的 XSTARS.xlam 或 `xstars_launch.scpt`，
  记下来并在 P0-5 处理。**

- [ ] **P0-5（仅当 P0-4 发现旧路径值）** 退出 Excel，手动删除那条 OPEN 值所在行对应的加载项文件，
  或等卸载阶段由 `uninstall.sh --apply` 统一清理后再复测。验收期间先记录现象即可，别让双载污染 V-02~V-05 的结论。

---

## V-01 Tkinter/ttkbootstrap 开箱可用（Blocker）

- [ ] ```bash
  "$HOME/Library/Application Support/XSTARS/python/bin/python3" -c "import ttkbootstrap as tb; root=tb.Window(); root.destroy(); print('TK_OK')"

  ```
  通过标准：打印 `TK_OK`，无 `Can't find a usable init.tcl`。
  结果：＿＿＿ 证据：＿＿＿
  若 Fail：记下完整报错（这就是 RK-01 实锤，触发 sitecustomize 缓解路线），**不要自行改文件**。

## V-02 首次未运行 Excel 时的目录预创建（Blocker）

> ⚠️ 偏离说明：你的账号此前跑过 Excel（dev 模式），「Excel 从未运行」前提已不成立——这在装 pkg 之前
> 就如此，不算 pkg 的错。按 Deviation 记录，验证重点改为「部署文件在 Excel 启动后仍在 + Ribbon 出现」。

- [ ] 启动 Excel（全新会话），确认 **XSTARS Ribbon 出现**（且只有一份，无重复 Ribbon——呼应 P0-4）。
- [ ] 再跑一次 P0-1 的 ls，确认 `XSTARS.xlam` 未被 Excel 启动过程删除/改名。
  结果：＿＿＿ 证据：＿＿＿

## V-03 INTERPRETER_MAC 沙盒读取（Blocker）

- [ ] 点 Ribbon 任一按钮触发 RunPython，在终端盯进程路径：

  ```bash
  pgrep -fl "Application Support/XSTARS/python/bin/python3"
  ```

  通过标准：进程命令行正是内置 python3 路径；Excel 无权限弹窗（若弹自动化/TCC 授权，允许后复测一次）。
  结果：＿＿＿ 证据：＿＿＿

## V-04 quarantine 放行路径（Blocker，macOS 15.3+ 行为）

- [ ] **用网络途径重新拿一份 pkg**（本机拷贝/U 盘不算——不带 quarantine）。可用：浏览器从 GitHub
  Actions 下载 artifact、AirDrop、微信文件传输助手等。验证隔离属性存在：

  ```bash
  xattr -p com.apple.quarantine /path/to/XSTARS-1.2.pkg
  ```

- [ ] 双击 pkg → 应被 Gatekeeper 拦 → 走「系统设置 → 隐私与安全性 → 仍要打开」+ 密码。
- [ ] 走完安装向导到成功（重复安装会 overlay 同路径，安全）。
  通过标准：放行成功进入安装向导并完成；**逐步截图**（这是要回填文档的实测勘误素材）。
  结果：＿＿＿ 证据：＿＿＿

## V-05 RunPython 反复触发稳定性（Blocker）

- [ ] 连续点 Ribbon 按钮 **5 次**执行测试任务，期间/结束后：

  ```bash
  pgrep -fl "XSTARS/python/bin/python3"   # 应随任务结束清空，无残留
  ps aux | grep -i "[e]xcel"              # Excel 无卡死
  ```

  通过标准：无僵尸 python 进程、无 Office 卡死。
  结果：＿＿＿ 证据：＿＿＿

---

## U-01 完整卸载（Blocker，破坏性——最后做）

- [ ] 退出 Excel 和 WPS（`pgrep -x "Microsoft Excel"` / `pgrep -x "WPS Office"` 应为空）。
- [ ] 先看 dry-run（默认，不改文件），**核对输出里每条 would 动作**：

  ```bash
  "$HOME/Library/Application Support/XSTARS/uninstall.sh"
  ```

- [ ] 确认无误后执行：

  ```bash
  "$HOME/Library/Application Support/XSTARS/uninstall.sh" --apply
  ```

- [ ] 检查点：
  - 输出含 `cleaned XSTARS Excel OPEN/OPENn values; integrity_check ok`；
  - 共享文件按 P0-2 的三态处理：有备份→`restored pre-install ...`；有 absent marker→删除；
    都没有→warning 且保持不动（这是 rev14 修复的 B 项，正好实测）；
  - 备份目录已生成：`~/Documents/XSTARS-uninstall-backups/uninstall-backup-<UTC>-<pid>/`。
- [ ] 启动 Excel **3 次**：每次都无「找不到加载项」弹窗；`~/.../Startup.localized/Excel/` 无 XSTARS.xlam。
- [ ] `pkgutil --pkg-info com.frank-sysu.xstars` 应报未找到。
  结果：＿＿＿ 证据：＿＿＿

## U-02 用户数据保全（Blocker）

- [ ] ```bash
  ls "$HOME/.xstars/"

  ```
  通过标准：`settings.json` 和图表重建产物**原样保留**（卸载不删用户数据）。
  结果：＿＿＿ 证据：＿＿＿

- [ ] **收尾**：确认 Excel 多次启动正常后再删 `~/Documents/XSTARS-uninstall-backups/`（文档建议保留）。

  然后**重装 pkg**恢复使用态。

---

## 回填 PR #6 的格式

每项一行：`<ID>: Pass|Fail|Deviation — <一句话证据，附截图/输出路径>`。
任何 Fail → 停止后续项，回对应 milestone 修复后再验（V-01 Fail → RK-01 缓解路线）。
全部 Pass 后 Draft PR #6 → Ready for review，M6/T6.2 闭合。
