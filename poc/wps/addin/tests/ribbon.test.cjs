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
  const context = {
    window: {
      alert: (message) => alerts.push(message),
      location: { origin: options.origin || "file://" },
      fetch: async (url, init) => {
        fetchCalls.push({ url, init });
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
      },
    },
  };

  vm.createContext(context);
  vm.runInContext(ribbonSource, context);
  return { alerts, context, fetchCalls };
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
    Value2: [[1, 2], [3, 4]],
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
        matrix: [[1, 2], [3, 4]],
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
      values: [[1, 2], [3, 4]],
    },
  });
  assert.deepEqual(JSON.parse(JSON.stringify(target.Value2)), [[1, 2], [3, 4]]);
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
