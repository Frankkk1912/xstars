const XSTARS_GATE0_PROBE_URL = "http://127.0.0.1:3891/probe";
const XSTARS_M04_ELISA_URL =
  "http://127.0.0.1:3892/probe/elisa-selection";
const XSTARS_M04_SHAPE_EXPORT_URL =
  "http://127.0.0.1:3892/probe/shape-export";
const XSTARS_M04_COM_PROBE_URL =
  "http://127.0.0.1:3892/probe/com-probe";
const XSTARS_M04_EXPORT_FORMATS = Object.freeze([
  "png",
  "tiff",
  "jpg",
  "pdf",
]);
const M04_TWO_STAGE_STATE = {
  standard: null,
  sample: null,
};

// M0.3 PoC：本机固定路径，仅用于 Gate 0 验证，不进入正式产品。
const XSTARS_GATE0_SERVICE = Object.freeze({
  baseUrl: "http://127.0.0.1:3892",
  pythonwPath: "C:\\Users\\daiyu\\miniforge3\\envs\\scrna\\pythonw.exe",
  scriptPath: "E:\\Documents\\GitHub\\xstars\\poc\\wps\\service_server.py",
  workDir: "E:\\Documents\\GitHub\\xstars\\poc\\wps",
  launchPollAttempts: 30,
  launchPollIntervalMs: 500,
  conflictWaitMs: 2000,
  dialogProbeCount: 3,
  dialogProbeIntervalMs: 700,
});

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function fetchService(url, init) {
  // 统一封装：永不抛出，返回可诊断的结果对象（status 0 表示连接层失败）。
  try {
    const response = await window.fetch(url, init);
    const text = await response.text();
    let payload = null;
    try {
      payload = JSON.parse(text);
    } catch (parseError) {
      return {
        ok: false,
        status: response.status,
        error: `响应不是有效 JSON：${parseError.message}`,
      };
    }
    return { ok: response.ok, status: response.status, payload };
  } catch (error) {
    return { ok: false, status: 0, error: error.message };
  }
}

async function checkServiceHealth() {
  const result = await fetchService(`${XSTARS_GATE0_SERVICE.baseUrl}/health`);
  return result.ok && result.payload && result.payload.ok
    ? result.payload
    : null;
}

function launchServiceViaShellExecute() {
  const application = window.Application;
  const oaAssist = application && application.OAAssist;
  if (!oaAssist || typeof oaAssist.ShellExecute !== "function") {
    throw new Error("OAAssist.ShellExecute 不可用，无法拉起本地服务");
  }
  // 官方签名只有 2 个参数：ShellExecute(Url, Params)，非 Windows API 5 参数版。
  // 传 5 个参数会被 WPS JSAPI 以 "too many parameters" 拒绝。
  return oaAssist.ShellExecute(
    XSTARS_GATE0_SERVICE.pythonwPath,
    `"${XSTARS_GATE0_SERVICE.scriptPath}" --port 3892`,
  );
}

function normalizeCellValue(value) {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  if (value === undefined) {
    return null;
  }
  return String(value);
}

function normalizeSelectionValues(value2, rowCount, columnCount) {
  if (!Number.isInteger(rowCount) || !Number.isInteger(columnCount)) {
    throw new Error("选区行列数无效");
  }
  if (rowCount < 1 || columnCount < 1 || rowCount > 200 || columnCount > 200) {
    throw new Error("选区必须是 1-200 行、1-200 列的连续区域");
  }

  let matrix;
  if (!Array.isArray(value2)) {
    matrix = [[value2]];
  } else if (Array.isArray(value2[0])) {
    matrix = value2;
  } else if (rowCount === 1) {
    matrix = [value2];
  } else if (columnCount === 1) {
    matrix = value2.map((value) => [value]);
  } else {
    throw new Error("WPS 返回了无法识别的二维选区数据");
  }

  if (
    matrix.length !== rowCount ||
    matrix.some((row) => !Array.isArray(row) || row.length !== columnCount)
  ) {
    throw new Error("WPS 选区尺寸与 Value2 数据不一致");
  }
  return matrix.map((row) => row.map(normalizeCellValue));
}

function getRangeAddress(range) {
  if (typeof range.Address === "function") {
    return range.Address();
  }
  return String(range.Address || "（未知）");
}

async function runM02Probe() {
  const application = window.Application;
  const selection = application.Selection;
  if (!selection || !selection.Rows || !selection.Columns) {
    throw new Error("请先选择一个连续的单元格区域");
  }
  if (selection.Areas && Number(selection.Areas.Count) !== 1) {
    throw new Error("M0.2 仅接受单个连续选区");
  }

  const rowCount = Number(selection.Rows.Count);
  const columnCount = Number(selection.Columns.Count);
  const values = normalizeSelectionValues(
    selection.Value2,
    rowCount,
    columnCount,
  );
  const requestBody = {
    selection: {
      address: getRangeAddress(selection),
      rows: rowCount,
      columns: columnCount,
      values,
    },
  };

  const response = await window.fetch(XSTARS_GATE0_PROBE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });
  const responseText = await response.text();
  let payload;
  try {
    payload = JSON.parse(responseText);
  } catch (error) {
    throw new Error(`探针返回的不是有效 JSON：${error.message}`);
  }
  if (!response.ok || !payload.ok) {
    const detail = payload.error
      ? `${payload.error.code}: ${payload.error.message}`
      : response.status;
    throw new Error(`回环探针请求失败：${detail}`);
  }

  const matrix = payload.matrix;
  if (!Array.isArray(matrix) || !Array.isArray(matrix[0])) {
    throw new Error("探针响应缺少二维写回矩阵");
  }
  const writeback = selection
    .Offset(0, columnCount + 1)
    .Resize(matrix.length, matrix[0].length);
  writeback.Value2 = matrix;

  const sheet = application.ActiveSheet;
  if (!sheet || !sheet.Shapes) {
    throw new Error("无法获取活动工作表的 Shapes 集合");
  }
  const imageAnchor = writeback.Offset(matrix.length + 1, 0);
  const picture = sheet.Shapes.AddPicture(
    payload.imagePath,
    0,
    -1,
    Number(imageAnchor.Left),
    Number(imageAnchor.Top),
    Number(payload.imageWidth),
    Number(payload.imageHeight),
  );

  const writebackAddress = getRangeAddress(writeback);
  window.alert(
    `M0.2 垂直链路成功\n选区：${requestBody.selection.address}\n写回：${writebackAddress}\n图片：${picture.Name || "已插入"}\nOrigin：${window.location.origin || "（未知）"}`,
  );
  return { payload, requestBody, writebackAddress, picture };
}

async function runM03ServiceStart() {
  const before = await checkServiceHealth();
  if (before) {
    window.alert(
      `M0.3 本地服务已在运行\nPID：${before.pid}\n已运行：${before.uptimeSeconds} 秒\n服务端记录 Origin：${before.requestOrigin}`,
    );
    return { before, after: before, shellResult: null, alreadyRunning: true };
  }

  const shellResult = launchServiceViaShellExecute();
  let launched = null;
  for (
    let attempt = 0;
    attempt < XSTARS_GATE0_SERVICE.launchPollAttempts;
    attempt += 1
  ) {
    await sleep(XSTARS_GATE0_SERVICE.launchPollIntervalMs);
    launched = await checkServiceHealth();
    if (launched) {
      break;
    }
  }
  if (!launched) {
    throw new Error(
      `ShellExecute 后 15 秒内服务未就绪（ShellExecute 返回：${String(shellResult)}）`,
    );
  }
  window.alert(
    `M0.3 服务拉起成功\nShellExecute 返回：${String(shellResult)}\nPID：${launched.pid}\n服务端记录 Origin：${launched.requestOrigin}`,
  );
  return { before: null, after: launched, shellResult, alreadyRunning: false };
}

async function runM03Dialog() {
  const dialogPromise = fetchService(`${XSTARS_GATE0_SERVICE.baseUrl}/dialog`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: "M0.3 Tkinter dialog" }),
  });

  // 对话框打开期间持续探活：服务必须保持响应，WPS 不得被阻塞。
  const healthLatencies = [];
  for (
    let index = 0;
    index < XSTARS_GATE0_SERVICE.dialogProbeCount;
    index += 1
  ) {
    await sleep(XSTARS_GATE0_SERVICE.dialogProbeIntervalMs);
    const startedAt = Date.now();
    const health = await checkServiceHealth();
    healthLatencies.push(health ? Date.now() - startedAt : null);
  }

  const result = await dialogPromise;
  if (!result.ok) {
    const detail = result.error
      ? result.error
      : result.payload && result.payload.error
        ? `${result.payload.error.code}: ${result.payload.error.message}`
        : `HTTP ${result.status}`;
    throw new Error(`Tkinter 对话框请求失败：${detail}`);
  }
  const unresponsive = healthLatencies.some((latency) => latency === null);
  window.alert(
    `M0.3 Tkinter 对话框完成\n选择：${result.payload.confirmed ? "确定" : "取消"}\n耗时：${result.payload.durationMs} ms\n对话框期间健康探活：${healthLatencies.map((latency) => (latency === null ? "无响应" : `${latency} ms`)).join("、")}\n${unresponsive ? "警告：对话框期间服务失去响应！" : "对话框打开期间服务保持响应，WPS 不应被冻结"}`,
  );
  return { payload: result.payload, healthLatencies };
}

async function runM03PortConflict() {
  const before = await checkServiceHealth();
  if (!before) {
    throw new Error("本地服务未运行，请先执行「拉起本地服务」");
  }

  const shellResult = launchServiceViaShellExecute();
  await sleep(XSTARS_GATE0_SERVICE.conflictWaitMs);
  const after = await checkServiceHealth();
  if (!after) {
    throw new Error(
      "发起第二实例后原服务失去响应——端口冲突影响了原服务，需立即排查",
    );
  }

  const diag = await fetchService(
    `${XSTARS_GATE0_SERVICE.baseUrl}/diagnostics`,
  );
  const conflictLines =
    diag.ok && diag.payload && Array.isArray(diag.payload.logTail)
      ? diag.payload.logTail.filter((line) => line.includes("PORT CONFLICT"))
      : [];
  const latestConflict =
    conflictLines[conflictLines.length - 1] || "（未在日志中找到冲突记录）";
  window.alert(
    `M0.3 端口冲突诊断\n第二实例 ShellExecute 返回：${String(shellResult)}\n原服务存活：PID ${before.pid} → PID ${after.pid}\n冲突记录：${latestConflict}`,
  );
  return { before, after, conflictLines, shellResult };
}

function rangeToM04Payload(range) {
  if (!range || !range.Rows || !range.Columns) {
    throw new Error("返回对象不是可读取的连续 Range");
  }
  const rows = Number(range.Rows.Count);
  const columns = Number(range.Columns.Count);
  return {
    address: getRangeAddress(range),
    values: normalizeSelectionValues(range.Value2, rows, columns),
  };
}

function m04ErrorDetail(result) {
  if (result.error) {
    return result.error;
  }
  if (result.payload && result.payload.error) {
    const error = result.payload.error;
    return `${error.code}: ${error.message}`;
  }
  if (result.payload && result.payload.code) {
    return `${result.payload.code}: ${result.payload.detail || "无详细信息"}`;
  }
  return `HTTP ${result.status}`;
}

async function postM04Selection(source, ranges) {
  const result = await fetchService(XSTARS_M04_ELISA_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, ranges }),
  });
  if (!result.ok || !result.payload || !result.payload.ok) {
    throw new Error(`ELISA 选区探针失败：${m04ErrorDetail(result)}`);
  }
  return result.payload;
}

async function runM04InputBoxProbe() {
  const application = window.Application;
  if (!application || typeof application.InputBox !== "function") {
    window.alert("M0.4 InputBox 探针不可用：Application.InputBox 不存在");
    return { unavailable: true };
  }

  let selectedRange;
  try {
    selectedRange = application.InputBox(
      "请用鼠标框选 ELISA 样本数据区域",
      "XSTARS M0.4 ELISA 选区探针",
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      8,
    );
  } catch (error) {
    window.alert(`M0.4 InputBox 探针异常：${error.message}`);
    return { error: error.message };
  }
  if (!selectedRange) {
    window.alert("M0.4 InputBox 探针：用户取消选区（非错误）");
    return { cancelled: true };
  }

  try {
    const range = rangeToM04Payload(selectedRange);
    const payload = await postM04Selection("inputbox", [range]);
    window.alert(
      `M0.4 InputBox 选区成功\n地址：${range.address}\n行列：${payload.ranges[0].rows} × ${payload.ranges[0].columns}\n非空：${payload.ranges[0].nonEmptyCells}`,
    );
    return { payload, range };
  } catch (error) {
    window.alert(`M0.4 InputBox 选区处理失败：${error.message}`);
    return { error: error.message };
  }
}

async function runM04TwoStageProbe() {
  try {
    if (!M04_TWO_STAGE_STATE.standard) {
      M04_TWO_STAGE_STATE.standard = rangeToM04Payload(
        window.Application.Selection,
      );
      window.alert(
        `M0.4 两阶段（1/3）：已记录标准品选区 ${M04_TWO_STAGE_STATE.standard.address}\n请框选样本区域后再次点击。`,
      );
      return { stage: "standard", range: M04_TWO_STAGE_STATE.standard };
    }
    if (!M04_TWO_STAGE_STATE.sample) {
      M04_TWO_STAGE_STATE.sample = rangeToM04Payload(
        window.Application.Selection,
      );
      window.alert(
        `M0.4 两阶段（2/3）：已记录样本选区 ${M04_TWO_STAGE_STATE.sample.address}\n再次点击以提交两个选区。`,
      );
      return { stage: "sample", range: M04_TWO_STAGE_STATE.sample };
    }

    const ranges = [M04_TWO_STAGE_STATE.standard, M04_TWO_STAGE_STATE.sample];
    const payload = await postM04Selection("two-stage", ranges);
    M04_TWO_STAGE_STATE.standard = null;
    M04_TWO_STAGE_STATE.sample = null;
    window.alert(
      `M0.4 两阶段（3/3）提交成功\n标准品：${ranges[0].address}\n样本：${ranges[1].address}`,
    );
    return { stage: "submitted", payload, ranges };
  } catch (error) {
    window.alert(`M0.4 两阶段选区失败：${error.message}`);
    return { error: error.message };
  }
}

async function runM04AddressFallback() {
  const addressInput = window.prompt(
    "请输入活动工作表中的 A1 地址（例如 C2:F10）",
    "",
  );
  if (addressInput === null) {
    window.alert("M0.4 地址兜底：用户取消（非错误）");
    return { cancelled: true };
  }
  const address = addressInput.trim();
  const addressPattern = /^\$?[A-Za-z]{1,3}\$?\d+(?::\$?[A-Za-z]{1,3}\$?\d+)?$/;
  if (!addressPattern.test(address)) {
    window.alert("M0.4 地址兜底：地址格式无效，请使用例如 C2:F10 的 A1 地址");
    return { invalid: true };
  }

  try {
    const sheet = window.Application.ActiveSheet;
    if (!sheet || typeof sheet.Range !== "function") {
      throw new Error("活动工作表不支持 Range(address)");
    }
    const range = rangeToM04Payload(sheet.Range(address));
    const payload = await postM04Selection("address", [range]);
    window.alert(`M0.4 地址兜底提交成功\n地址：${range.address}`);
    return { payload, range };
  } catch (error) {
    window.alert(`M0.4 地址兜底失败：${error.message}`);
    return { error: error.message };
  }
}

function selectedShape(selection) {
  if (!selection) {
    throw new Error("请先选择一张图片或 Shape");
  }
  let shapeRange = null;
  try {
    shapeRange = selection.ShapeRange || null;
  } catch {
    shapeRange = null;
  }
  if (shapeRange) {
    if (typeof shapeRange.Item === "function") {
      return shapeRange.Item(1);
    }
    if (typeof shapeRange.CopyPicture === "function") {
      return shapeRange;
    }
  }
  if (typeof selection.CopyPicture === "function") {
    return selection;
  }
  throw new Error("当前 Selection 不包含可复制的 ShapeRange");
}

// WPS 内嵌浏览器不支持 window.prompt（静默返回 null）；宿主原生 InputBox（Type=2 文本）
// 已在 M0.4 实机步骤 1 验证可用，故优先使用；window.prompt 仅作为非 WPS 环境回退。
function m04PromptText(message, title, defaultValue) {
  try {
    const application = window.Application;
    if (application && typeof application.InputBox === "function") {
      const result = application.InputBox(message, title, defaultValue, undefined, undefined, undefined, undefined, 2);
      if (result === null || result === false) {
        return { cancelled: true, value: null };
      }
      const text = String(result).trim();
      return { cancelled: false, value: text === "" ? defaultValue : text };
    }
  } catch (error) {
    // fall through to window.prompt
  }
  if (typeof window.prompt === "function") {
    const result = window.prompt(message, defaultValue);
    if (result === null) {
      return { cancelled: true, value: null };
    }
    const text = String(result).trim();
    return { cancelled: false, value: text === "" ? defaultValue : text };
  }
  return { cancelled: false, value: defaultValue };
}

async function runM04ShapeExportProbe() {
  const selection = window.Application.Selection;
  let shape;
  try {
    shape = selectedShape(selection);
  } catch (error) {
    window.alert(`M0.4 Shape 导出失败：${error.message}`);
    return { error: error.message };
  }

  const formatResult = m04PromptText("导出格式：png/tiff/jpg/pdf", "M0.4 Shape 导出", "png");
  if (formatResult.cancelled) {
    window.alert("M0.4 Shape 导出：用户取消（非错误）");
    return { cancelled: true };
  }
  const format = formatResult.value.toLowerCase();
  const dpiResult = m04PromptText("目标 DPI（72-1200）", "M0.4 Shape 导出", "300");
  if (dpiResult.cancelled) {
    window.alert("M0.4 Shape 导出：用户取消（非错误）");
    return { cancelled: true };
  }
  const dpi = Number(dpiResult.value);
  if (!XSTARS_M04_EXPORT_FORMATS.includes(format)) {
    window.alert(`M0.4 Shape 导出失败：不支持格式 ${format}`);
    return { invalid: true };
  }
  if (!Number.isInteger(dpi) || dpi < 72 || dpi > 1200) {
    window.alert("M0.4 Shape 导出失败：DPI 必须是 72-1200 的整数");
    return { invalid: true };
  }

  let copyMode = "xlPrinter/xlPicture";
  try {
    shape.CopyPicture(2, -4147);
  } catch {
    try {
      shape.CopyPicture(1, 2);
      copyMode = "xlScreen/xlBitmap fallback";
    } catch (error) {
      window.alert(`M0.4 Shape CopyPicture 失败：${error.message}`);
      return { error: error.message };
    }
  }

  // WPS CopyPicture returns synchronously, but the Windows clipboard can lag
  // briefly before the DIB/EMF becomes visible to another process.
  await sleep(150);
  const exportResult = await fetchService(XSTARS_M04_SHAPE_EXPORT_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ format, dpi }),
  });
  if (!exportResult.ok || !exportResult.payload || !exportResult.payload.ok) {
    const detail = m04ErrorDetail(exportResult);
    window.alert(`M0.4 Shape 导出失败：${detail}`);
    return { error: detail };
  }

  const comResult = await fetchService(XSTARS_M04_COM_PROBE_URL);
  const comSummary =
    comResult.payload && comResult.payload.ok
      ? `可用（${comResult.payload.progId} ${comResult.payload.version || ""}）`
      : `不可用（${m04ErrorDetail(comResult)}）`;
  const payload = exportResult.payload;
  const selectionType =
    selection && selection.Type !== undefined
      ? String(selection.Type)
      : "（未暴露 Type）";
  window.alert(
    `M0.4 Shape 导出成功\nSelection.Type：${selectionType}\nCopyPicture：${copyMode}\n文件：${payload.outputPath}\n像素：${payload.width} × ${payload.height}\nDPI：${payload.dpi}\nCOM Ket.Application：${comSummary}`,
  );
  return {
    payload,
    copyMode,
    com: comResult.payload,
    selectionType,
  };
}

function reportM03Error(title, error) {
  window.alert(`${title}\n${error.message}`);
}

function OnAddinLoad(ribbonUI) {
  window.Application.ribbonUI = ribbonUI;
  return true;
}

function OnAction(control) {
  if (control.Id === "xstarsGate0Callback") {
    const workbook = window.Application.ActiveWorkbook;
    const workbookName = workbook ? workbook.Name : "（无活动工作簿）";
    const origin = window.location.origin || "（未知）";
    window.alert(
      `XSTARS Gate 0 回调成功\n工作簿：${workbookName}\nOrigin：${origin}`,
    );
    return true;
  }
  if (control.Id === "xstarsM02Probe") {
    void runM02Probe().catch((error) => {
      window.alert(`M0.2 垂直链路失败\n${error.message}`);
    });
  }
  if (control.Id === "xstarsM03ServiceStart") {
    void runM03ServiceStart().catch((error) => {
      reportM03Error("M0.3 服务拉起失败", error);
    });
  }
  if (control.Id === "xstarsM03Dialog") {
    void runM03Dialog().catch((error) => {
      reportM03Error("M0.3 对话框测试失败", error);
    });
  }
  if (control.Id === "xstarsM03PortConflict") {
    void runM03PortConflict().catch((error) => {
      reportM03Error("M0.3 端口冲突诊断失败", error);
    });
  }
  if (control.Id === "xstarsM04InputBox") {
    void runM04InputBoxProbe();
  }
  if (control.Id === "xstarsM04TwoStage") {
    void runM04TwoStageProbe();
  }
  if (control.Id === "xstarsM04ShapeExport") {
    void runM04ShapeExportProbe();
  }
  return true;
}

function GetImage() {
  return "images/1.svg";
}

// WPS resolves these Ribbon callbacks by their global names.
window.OnAddinLoad = OnAddinLoad;
window.OnAction = OnAction;
window.GetImage = GetImage;
window.XstarsGate0 = Object.freeze({
  normalizeSelectionValues,
  runM02Probe,
  checkServiceHealth,
  launchServiceViaShellExecute,
  runM03ServiceStart,
  runM03Dialog,
  runM03PortConflict,
  runM04InputBoxProbe,
  runM04TwoStageProbe,
  runM04AddressFallback,
  runM04ShapeExportProbe,
});
