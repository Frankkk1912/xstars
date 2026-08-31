const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const ribbonSource = fs.readFileSync(
  path.join(__dirname, "..", "js", "ribbon.js"),
  "utf8",
);

function loadRibbon(options = {}) {
  const alerts = [];
  const fetchCalls = [];
  const shellCalls = [];
  const promptCalls = [];
  const promptResponses = [...(options.promptResponses || [])];
  const context = {
    setTimeout,
    window: {
      alert: (message) => alerts.push(message),
      prompt: (message, defaultValue) => {
        promptCalls.push({ message, defaultValue });
        return options.promptHandler
          ? options.promptHandler(message, defaultValue)
          : promptResponses.shift();
      },
      location: { origin: options.origin || "file://" },
      fetch: async (url, init) => {
        fetchCalls.push({ url, init });
        if (options.fetchHandler) {
          return await options.fetchHandler(url, init);
        }
        if (!options.response) {
          throw new Error("fetch response not configured");
        }
        return {
          ok: options.response.ok !== false,
          status: options.response.status || 200,
          text: async () => JSON.stringify(options.response.body),
        };
      },
      Application: {
        ActiveWorkbook: options.workbookName
          ? { Name: options.workbookName }
          : null,
        ActiveSheet: options.activeSheet || null,
        Selection: options.selection || null,
        InputBox: options.inputBox,
        OAAssist: options.oaAssist || null,
      },
    },
  };

  vm.createContext(context);
  vm.runInContext(ribbonSource, context);
  return { alerts, context, fetchCalls, shellCalls, promptCalls };
}

test("OnAddinLoad stores the WPS Ribbon object", () => {
  const { context } = loadRibbon();
  const ribbonUI = { Invalidate: () => {} };

  assert.equal(context.OnAddinLoad(ribbonUI), true);
  assert.equal(context.window.Application.ribbonUI, ribbonUI);
});

test("Gate 0 button proves the callback and reports the workbook", () => {
  const { alerts, context } = loadRibbon({ workbookName: "gate0.xlsx" });

  assert.equal(context.OnAction({ Id: "xstarsGate0Callback" }), true);
  assert.deepEqual(alerts, [
    "XSTARS Gate 0 回调成功\n工作簿：gate0.xlsx\nOrigin：file://",
  ]);
});

test("selection normalization handles a one-column Value2 array", () => {
  const { context } = loadRibbon();

  const matrix = context.window.XstarsGate0.normalizeSelectionValues(
    [1, 2, 3],
    3,
    1,
  );
  assert.deepEqual(JSON.parse(JSON.stringify(matrix)), [[1], [2], [3]]);
});

test("M0.2 sends Selection, writes Value2, and embeds the returned PNG", async () => {
  const pictureCalls = [];
  const target = {
    Value2: null,
    Address: () => "$D$1:$E$2",
    Offset: (rows, columns) => {
      assert.equal(rows, 3);
      assert.equal(columns, 0);
      return { Left: 480, Top: 120 };
    },
  };
  const selection = {
    Address: () => "$A$1:$B$2",
    Areas: { Count: 1 },
    Rows: { Count: 2 },
    Columns: { Count: 2 },
    Value2: [
      [1, 2],
      [3, 4],
    ],
    Offset: (rows, columns) => {
      assert.equal(rows, 0);
      assert.equal(columns, 3);
      return {
        Resize: (resizeRows, resizeColumns) => {
          assert.equal(resizeRows, 2);
          assert.equal(resizeColumns, 2);
          return target;
        },
      };
    },
  };
  const activeSheet = {
    Shapes: {
      AddPicture: (...args) => {
        pictureCalls.push(args);
        return { Name: "Picture 1" };
      },
    },
  };
  const { alerts, context, fetchCalls } = loadRibbon({
    selection,
    activeSheet,
    response: {
      body: {
        ok: true,
        matrix: [
          [1, 2],
          [3, 4],
        ],
        imagePath: "C:\\Temp\\xstars-gate0-probe.png",
        imageWidth: 320,
        imageHeight: 180,
      },
    },
  });

  const result = await context.window.XstarsGate0.runM02Probe();

  assert.equal(fetchCalls[0].url, "http://127.0.0.1:3891/probe");
  assert.deepEqual(JSON.parse(fetchCalls[0].init.body), {
    selection: {
      address: "$A$1:$B$2",
      rows: 2,
      columns: 2,
      values: [
        [1, 2],
        [3, 4],
      ],
    },
  });
  assert.deepEqual(JSON.parse(JSON.stringify(target.Value2)), [
    [1, 2],
    [3, 4],
  ]);
  assert.deepEqual(pictureCalls, [
    ["C:\\Temp\\xstars-gate0-probe.png", 0, -1, 480, 120, 320, 180],
  ]);
  assert.equal(result.writebackAddress, "$D$1:$E$2");
  assert.match(alerts[0], /M0\.2 垂直链路成功/);
});

test("Unrelated controls are ignored", () => {
  const { alerts, context } = loadRibbon();

  assert.equal(context.OnAction({ Id: "otherControl" }), true);
  assert.deepEqual(alerts, []);
});

function jsonResponse(body, ok = true, status = 200) {
  return {
    ok,
    status,
    text: async () => JSON.stringify(body),
  };
}

function serviceHealthHandler(failHealthUntil = 0) {
  let healthCount = 0;
  return (url) => {
    if (url.endsWith("/health")) {
      healthCount += 1;
      if (healthCount <= failHealthUntil) {
        throw new Error("connection refused");
      }
      return jsonResponse({
        ok: true,
        service: "xstars-wps-gate0-service",
        port: 3892,
        pid: 4242,
        uptimeSeconds: 0.5,
        requestOrigin: "null",
      });
    }
    if (url.endsWith("/diagnostics")) {
      return jsonResponse({
        ok: true,
        logTail: [
          "2026-08-29T17:20:00 pid=999 PORT CONFLICT on 127.0.0.1:3892 while starting second instance: OSError: [WinError 10048]",
        ],
        requestOrigin: "null",
      });
    }
    return jsonResponse({ ok: false }, false, 404);
  };
}

test("M0.3 launches the service via ShellExecute when health fails", async () => {
  const handler = serviceHealthHandler(2); // 失败两次后第三次成功
  const { alerts, context, fetchCalls } = loadRibbon({
    fetchHandler: handler,
    oaAssist: {
      ShellExecute: (...args) => {
        assert.equal(args.length, 2);
        assert.equal(
          args[0],
          "C:\\Users\\daiyu\\miniforge3\\envs\\scrna\\pythonw.exe",
        );
        assert.match(args[1], /service_server\.py" --port 3892$/);
        return 42;
      },
    },
  });

  const result = await context.window.XstarsGate0.runM03ServiceStart();

  assert.equal(result.alreadyRunning, false);
  assert.equal(result.shellResult, 42);
  assert.equal(result.after.pid, 4242);
  assert.match(alerts[0], /M0\.3 服务拉起成功/);
  assert.match(alerts[0], /Origin：null/);
  assert.ok(fetchCalls.some((call) => call.url.endsWith("/health")));
});

test("M0.3 reports an already-running service without ShellExecute", async () => {
  const shellCalls = [];
  const { alerts, context } = loadRibbon({
    fetchHandler: serviceHealthHandler(0),
    oaAssist: {
      ShellExecute: (...args) => {
        shellCalls.push(args);
        return 1;
      },
    },
  });

  const result = await context.window.XstarsGate0.runM03ServiceStart();

  assert.equal(result.alreadyRunning, true);
  assert.deepEqual(shellCalls, []);
  assert.match(alerts[0], /已在运行/);
});

test("M0.3 fails with a diagnostic when ShellExecute is unavailable", async () => {
  const { alerts, context } = loadRibbon({
    fetchHandler: serviceHealthHandler(999),
  });

  await assert.rejects(
    context.window.XstarsGate0.runM03ServiceStart(),
    /OAAssist\.ShellExecute 不可用/,
  );
  assert.deepEqual(alerts, []);
});

test("M0.3 dialog stays responsive and reports the choice", async () => {
  const healthLatencies = [];
  const handler = (url, init) => {
    if (url.endsWith("/dialog")) {
      assert.equal(init.method, "POST");
      return jsonResponse({
        ok: true,
        confirmed: false,
        durationMs: 2500,
        requestOrigin: "null",
      });
    }
    if (url.endsWith("/health")) {
      healthLatencies.push(12);
      return jsonResponse({ ok: true, pid: 4242 });
    }
    return jsonResponse({ ok: false }, false, 404);
  };
  const { alerts, context, fetchCalls } = loadRibbon({ fetchHandler: handler });

  const result = await context.window.XstarsGate0.runM03Dialog();

  assert.equal(result.payload.confirmed, false);
  assert.ok(fetchCalls.some((call) => call.url.endsWith("/dialog")));
  assert.match(alerts[0], /M0\.3 Tkinter 对话框完成/);
  assert.match(alerts[0], /选择：取消/);
  assert.match(alerts[0], /服务保持响应/);
});

test("M0.3 port conflict keeps the first instance alive and surfaces the log", async () => {
  const shellCalls = [];
  const healthPids = [1111, 1111];
  let healthIndex = 0;
  const handler = (url) => {
    if (url.endsWith("/health")) {
      const pid = healthPids[Math.min(healthIndex, healthPids.length - 1)];
      healthIndex += 1;
      return jsonResponse({ ok: true, pid, uptimeSeconds: 1 });
    }
    if (url.endsWith("/diagnostics")) {
      return jsonResponse({
        ok: true,
        logTail: [
          "2026-08-29T17:20:00 pid=999 PORT CONFLICT on 127.0.0.1:3892 while starting second instance: OSError: [WinError 10048]",
        ],
      });
    }
    return jsonResponse({ ok: false }, false, 404);
  };
  const { alerts, context } = loadRibbon({
    fetchHandler: handler,
    oaAssist: {
      ShellExecute: (...args) => {
        shellCalls.push(args);
        return 7;
      },
    },
  });

  const result = await context.window.XstarsGate0.runM03PortConflict();

  assert.equal(shellCalls.length, 1);
  assert.equal(result.before.pid, 1111);
  assert.equal(result.after.pid, 1111);
  assert.equal(result.conflictLines.length, 1);
  assert.match(alerts[0], /M0\.3 端口冲突诊断/);
  assert.match(alerts[0], /PID 1111 → PID 1111/);
  assert.match(alerts[0], /PORT CONFLICT/);
});

test("M0.3 port conflict refuses to run when the service is down", async () => {
  const { context } = loadRibbon({
    fetchHandler: serviceHealthHandler(999),
    oaAssist: { ShellExecute: () => 1 },
  });

  await assert.rejects(
    context.window.XstarsGate0.runM03PortConflict(),
    /本地服务未运行/,
  );
});

function makeRange(address, values) {
  return {
    Address: () => address,
    Rows: { Count: values.length },
    Columns: { Count: values[0].length },
    Value2: values,
  };
}

function m04SelectionResponse(body) {
  return jsonResponse({
    ok: true,
    source: body.source,
    ranges: body.ranges.map((range) => ({
      ...range,
      rows: range.values.length,
      columns: range.values[0].length,
      nonEmptyCells: range.values.flat().filter((value) => value != null).length,
    })),
  });
}

test("M0.4 InputBox probe sends the selected Range as a bounded matrix", async () => {
  const inputBoxCalls = [];
  const selectedRange = makeRange("$C$2:$D$3", [
    [1, 2],
    [3, null],
  ]);
  const { alerts, context, fetchCalls } = loadRibbon({
    inputBox: (...args) => {
      inputBoxCalls.push(args);
      return selectedRange;
    },
    fetchHandler: (_url, init) =>
      m04SelectionResponse(JSON.parse(init.body)),
  });

  const result = await context.window.XstarsGate0.runM04InputBoxProbe();

  assert.equal(inputBoxCalls.length, 1);
  assert.equal(inputBoxCalls[0].length, 8);
  assert.equal(inputBoxCalls[0][7], 8);
  assert.equal(fetchCalls[0].url, "http://127.0.0.1:3892/probe/elisa-selection");
  assert.deepEqual(JSON.parse(fetchCalls[0].init.body), {
    source: "inputbox",
    ranges: [
      {
        address: "$C$2:$D$3",
        values: [
          [1, 2],
          [3, null],
        ],
      },
    ],
  });
  assert.equal(result.payload.ranges[0].nonEmptyCells, 3);
  assert.match(alerts[0], /InputBox 选区成功/);
});

test("M0.4 InputBox cancellation is reported without a failed request", async () => {
  const { alerts, context, fetchCalls } = loadRibbon({
    inputBox: () => null,
  });

  const result = await context.window.XstarsGate0.runM04InputBoxProbe();

  assert.equal(result.cancelled, true);
  assert.equal(fetchCalls.length, 0);
  assert.match(alerts[0], /用户取消选区（非错误）/);
});

test("M0.4 two-stage probe records two selections then submits on third click", async () => {
  const standard = makeRange("$A$1:$B$2", [
    [1, 2],
    [3, 4],
  ]);
  const sample = makeRange("$D$1:$D$2", [[5], [6]]);
  const { alerts, context, fetchCalls } = loadRibbon({
    selection: standard,
    fetchHandler: (_url, init) =>
      m04SelectionResponse(JSON.parse(init.body)),
  });

  const first = await context.window.XstarsGate0.runM04TwoStageProbe();
  context.window.Application.Selection = sample;
  const second = await context.window.XstarsGate0.runM04TwoStageProbe();
  const third = await context.window.XstarsGate0.runM04TwoStageProbe();

  assert.equal(first.stage, "standard");
  assert.equal(second.stage, "sample");
  assert.equal(third.stage, "submitted");
  assert.equal(fetchCalls.length, 1);
  assert.deepEqual(JSON.parse(fetchCalls[0].init.body), {
    source: "two-stage",
    ranges: [
      { address: "$A$1:$B$2", values: [[1, 2], [3, 4]] },
      { address: "$D$1:$D$2", values: [[5], [6]] },
    ],
  });
  assert.match(alerts[0], /（1\/3）/);
  assert.match(alerts[1], /（2\/3）/);
  assert.match(alerts[2], /（3\/3）提交成功/);
});

test("M0.4 address fallback reads a valid A1 range", async () => {
  const requestedAddresses = [];
  const range = makeRange("$C$2:$F$3", [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
  ]);
  const { context, fetchCalls } = loadRibbon({
    activeSheet: {
      Range: (address) => {
        requestedAddresses.push(address);
        return range;
      },
    },
    promptResponses: ["C2:F3"],
    fetchHandler: (_url, init) =>
      m04SelectionResponse(JSON.parse(init.body)),
  });

  const result = await context.window.XstarsGate0.runM04AddressFallback();

  assert.deepEqual(requestedAddresses, ["C2:F3"]);
  assert.equal(JSON.parse(fetchCalls[0].init.body).source, "address");
  assert.equal(result.range.address, "$C$2:$F$3");
});

test("M0.4 address fallback rejects an invalid address without reading the sheet", async () => {
  const rangeCalls = [];
  const { alerts, context, fetchCalls } = loadRibbon({
    activeSheet: { Range: (...args) => rangeCalls.push(args) },
    promptResponses: ["not an address"],
  });

  const result = await context.window.XstarsGate0.runM04AddressFallback();

  assert.equal(result.invalid, true);
  assert.equal(rangeCalls.length, 0);
  assert.equal(fetchCalls.length, 0);
  assert.match(alerts[0], /地址格式无效/);
});

test("M0.4 Shape export uses printer-picture copy and probes COM", async () => {
  const copyCalls = [];
  const shape = { CopyPicture: (...args) => copyCalls.push(args) };
  const selection = {
    Type: "ShapeRange",
    ShapeRange: { Item: (index) => (index === 1 ? shape : null) },
  };
  const handler = (url, init) => {
    if (url.endsWith("/probe/shape-export")) {
      assert.deepEqual(JSON.parse(init.body), { format: "png", dpi: 300 });
      return jsonResponse({
        ok: true,
        outputPath: "C:\\Temp\\shape.png",
        width: 1200,
        height: 800,
        dpi: 300,
      });
    }
    if (url.endsWith("/probe/com-probe")) {
      return jsonResponse({
        ok: true,
        progId: "Ket.Application",
        version: "12.1.0",
      });
    }
    return jsonResponse({ ok: false }, false, 404);
  };
  const { alerts, context, fetchCalls } = loadRibbon({
    selection,
    promptResponses: ["png", "300"],
    fetchHandler: handler,
  });

  const result = await context.window.XstarsGate0.runM04ShapeExportProbe();

  assert.deepEqual(copyCalls, [[2, -4147]]);
  assert.equal(fetchCalls.length, 2);
  assert.equal(result.copyMode, "xlPrinter/xlPicture");
  assert.equal(result.com.ok, true);
  assert.match(alerts[0], /Selection\.Type：ShapeRange/);
  assert.match(alerts[0], /COM Ket\.Application：可用/);
});

test("M0.4 Shape export prefers the host InputBox over window.prompt for format/DPI", async () => {
  const shape = { CopyPicture: () => {} };
  const selection = {
    Type: "ShapeRange",
    ShapeRange: { Item: (index) => (index === 1 ? shape : null) },
  };
  const handler = (url, init) => {
    if (url.endsWith("/probe/shape-export")) {
      assert.deepEqual(JSON.parse(init.body), { format: "png", dpi: 600 });
      return jsonResponse({
        ok: true,
        outputPath: "C:\\Temp\\shape600.png",
        width: 2400,
        height: 1600,
        dpi: 600,
      });
    }
    if (url.endsWith("/probe/com-probe")) {
      return jsonResponse(
        { ok: false, error: { code: "COM_UNAVAILABLE", message: "无效的类字符串" } },
        false,
        200,
      );
    }
    return jsonResponse({ ok: false }, false, 404);
  };
  const inputBoxCalls = [];
  const inputBox = (...args) => {
    inputBoxCalls.push(args);
    return inputBoxCalls.length === 1 ? "png" : "600";
  };
  const { alerts, fetchCalls, context } = loadRibbon({
    selection,
    inputBox,
    fetchHandler: handler,
  });

  const result = await context.window.XstarsGate0.runM04ShapeExportProbe();

  assert.equal(inputBoxCalls.length, 2);
  assert.equal(inputBoxCalls[0][0], "导出格式：png/tiff/jpg/pdf");
  assert.equal(inputBoxCalls[0][7], 2);
  assert.equal(inputBoxCalls[1][0], "目标 DPI（72-1200）");
  assert.deepEqual(JSON.parse(fetchCalls[0].init.body), { format: "png", dpi: 600 });
  assert.equal(result.copyMode, "xlPrinter/xlPicture");
  assert.match(alerts[0], /COM Ket\.Application：不可用/);
});

test("M0.4 Shape export treats host InputBox cancellation as non-error", async () => {
  const shape = { CopyPicture: () => {} };
  const selection = {
    Type: "ShapeRange",
    ShapeRange: { Item: (index) => (index === 1 ? shape : null) },
  };
  const { alerts, fetchCalls, context } = loadRibbon({
    selection,
    inputBox: () => null,
    fetchHandler: () => jsonResponse({ ok: false }, false, 404),
  });

  const result = await context.window.XstarsGate0.runM04ShapeExportProbe();

  assert.equal(result.cancelled, true);
  assert.equal(fetchCalls.length, 0);
  assert.match(alerts[0], /用户取消（非错误）/);
});
