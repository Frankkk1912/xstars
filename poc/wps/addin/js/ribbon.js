const XSTARS_GATE0_PROBE_URL = "http://127.0.0.1:3891/probe";

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
  const values = normalizeSelectionValues(selection.Value2, rowCount, columnCount);
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
    const detail = payload.error ? `${payload.error.code}: ${payload.error.message}` : response.status;
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
  return true;
}

function GetImage() {
  return "images/1.svg";
}

// WPS resolves these Ribbon callbacks by their global names.
window.OnAddinLoad = OnAddinLoad;
window.OnAction = OnAction;
window.GetImage = GetImage;
window.XstarsGate0 = Object.freeze({ normalizeSelectionValues, runM02Probe });
