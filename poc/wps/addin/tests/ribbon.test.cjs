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
  const context = {
    setTimeout,
    window: {
      alert: (message) => alerts.push(message),
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
        OAAssist: options.oaAssist || null,
      },
    },
  };

  vm.createContext(context);
  vm.runInContext(ribbonSource, context);
  return { alerts, context, fetchCalls, shellCalls };
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
        assert.equal(args[0], "C:\\Users\\daiyu\\miniforge3\\envs\\scrna\\pythonw.exe");
        assert.match(args[1], /service_server\.py" --port 3892$/);
        assert.equal(args[3], "open");
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
