(function exposeSpreadsheet(root) {

  const MAX_ROWS = 200;
  const MAX_COLUMNS = 200;

  function normalizeCellValue(value) {
    if (value === undefined || value === null) {
      return null;
    }
    if (typeof value === "number") {
      if (!Number.isFinite(value)) {
        throw new Error("选区包含无效数字");
      }
      return value;
    }
    if (typeof value === "string" || typeof value === "boolean") {
      return value;
    }
    return String(value);
  }

  function normalizeSelectionValues(value2, rowCount, columnCount) {
    if (!Number.isInteger(rowCount) || !Number.isInteger(columnCount)) {
      throw new Error("选区行列数无效");
    }
    if (
      rowCount < 1 ||
      columnCount < 1 ||
      rowCount > MAX_ROWS ||
      columnCount > MAX_COLUMNS
    ) {
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
    const address = typeof range.Address === "function"
      ? range.Address()
      : range.Address;
    if (typeof address !== "string" || address.trim() === "") {
      throw new Error("无法读取选区地址");
    }
    return address.trim();
  }

  function readRange(application, range) {
    if (!application || !range || !range.Rows || !range.Columns) {
      throw new Error("请先选择一个连续的单元格区域");
    }
    if (range.Areas && Number(range.Areas.Count) !== 1) {
      throw new Error("仅支持单个连续选区");
    }
    const sheet = range.Worksheet || application.ActiveSheet;
    if (!sheet || typeof sheet.Name !== "string" || sheet.Name.trim() === "") {
      throw new Error("无法读取活动工作表名称");
    }
    const rowCount = Number(range.Rows.Count);
    const columnCount = Number(range.Columns.Count);
    return {
      version: "1.0",
      values: normalizeSelectionValues(range.Value2, rowCount, columnCount),
      address: getRangeAddress(range),
      sheet: sheet.Name,
    };
  }

  function readSelection(application) {
    if (!application) {
      throw new Error("无法访问 WPS 表格应用程序");
    }
    return readRange(application, application.Selection);
  }

  function promptRange(application, message, title) {
    if (!application || typeof application.InputBox !== "function") {
      throw new Error("当前 WPS 不支持 InputBox 选区，请升级到受支持版本");
    }
    let selected;
    try {
      selected = application.InputBox(
        message,
        title,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        8,
      );
    } catch {
      return null;
    }
    return selected ? readRange(application, selected) : null;
  }

  function promptText(application, message, title, defaultValue) {
    if (!application || typeof application.InputBox !== "function") {
      throw new Error("当前 WPS 不支持参数输入");
    }
    let value;
    try {
      value = application.InputBox(
        message,
        title,
        defaultValue,
        undefined,
        undefined,
        undefined,
        undefined,
        2,
      );
    } catch {
      return null;
    }
    if (value === null || value === false) {
      return null;
    }
    const text = String(value).trim();
    return text || defaultValue;
  }

  function selectedShape(application) {
    const selection = application && application.Selection;
    if (!selection) {
      throw new Error("请先选择一张图片或 Shape");
    }
    let shapeRange = null;
    try {
      shapeRange = selection.ShapeRange || null;
    } catch {
      shapeRange = null;
    }
    if (shapeRange && typeof shapeRange.Item === "function") {
      return shapeRange.Item(1);
    }
    if (shapeRange && typeof shapeRange.CopyPicture === "function") {
      return shapeRange;
    }
    if (typeof selection.CopyPicture === "function") {
      return selection;
    }
    throw new Error("当前 Selection 不包含可导出的图片");
  }

  function requireMatrix(values, label) {
    if (
      !Array.isArray(values) ||
      values.length === 0 ||
      !Array.isArray(values[0]) ||
      values[0].length === 0
    ) {
      throw new Error(`${label} 缺少二维写回数据`);
    }
    const width = values[0].length;
    if (values.some((row) => !Array.isArray(row) || row.length !== width)) {
      throw new Error(`${label} 写回数据不是矩形`);
    }
    return { rows: values.length, columns: width };
  }

  function setStatus(application, message) {
    if (application) {
      application.StatusBar = message || false;
    }
  }

  function showError(application, message) {
    const text = String(message || "XSTARS 操作失败");
    setStatus(application, `XSTARS: 失败 — ${text}`);
    if (typeof root.alert === "function") {
      root.alert(text);
    }
  }

  function executeWritebackPlan(application, plan) {
    if (!plan || typeof plan !== "object") {
      throw new Error("服务响应缺少 WritebackPlan");
    }
    if (plan.version !== "1.0") {
      throw new Error("WritebackPlan 版本不受支持");
    }
    const sheet = application && application.ActiveSheet;
    if (!sheet || typeof sheet.Range !== "function") {
      throw new Error("无法访问活动工作表");
    }

    const writtenRanges = [];
    for (const table of plan.tables || []) {
      if (!table || typeof table.startCell !== "string") {
        throw new Error("表格写回缺少起始单元格");
      }
      const size = requireMatrix(table.values, "表格");
      const target = sheet.Range(table.startCell).Resize(size.rows, size.columns);
      target.Value2 = table.values;
      writtenRanges.push(target);
    }

    const pictures = [];
    for (const image of plan.images || []) {
      if (
        !image ||
        typeof image.anchorCell !== "string" ||
        !image.artifact ||
        typeof image.artifact.path !== "string"
      ) {
        throw new Error("图片写回缺少锚点或本地产物路径");
      }
      if (!sheet.Shapes || typeof sheet.Shapes.AddPicture !== "function") {
        throw new Error("活动工作表不支持图片插入");
      }
      const anchor = sheet.Range(image.anchorCell);
      const width = image.width == null ? -1 : Number(image.width);
      const height = image.height == null ? -1 : Number(image.height);
      if (
        (width !== -1 && (!Number.isFinite(width) || width <= 0)) ||
        (height !== -1 && (!Number.isFinite(height) || height <= 0))
      ) {
        throw new Error("图片尺寸无效");
      }
      const picture = sheet.Shapes.AddPicture(
        image.artifact.path,
        0,
        -1,
        Number(anchor.Left),
        Number(anchor.Top),
        width,
        height,
      );
      const requestedName = typeof image.pictureId === "string" && image.pictureId
        ? image.pictureId
        : image.name;
      if (picture && typeof requestedName === "string" && requestedName.trim()) {
        try {
          picture.Name = requestedName;
        } catch {
          // WPS builds that reject Shape.Name assignment keep the native name.
          // Analysis writeback remains valid; export will use clipboard fallback.
        }
      }
      pictures.push(picture);
    }

    setStatus(application, plan.statusMessage || "XSTARS: 完成");
    return { writtenRanges, pictures };
  }

  root.XstarsSpreadsheet = Object.freeze({
    executeWritebackPlan,
    getRangeAddress,
    normalizeCellValue,
    normalizeSelectionValues,
    promptRange,
    promptText,
    readRange,
    readSelection,
    selectedShape,
    setStatus,
    showError,
  });
})(typeof window === "undefined" ? globalThis : window);
