const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "spreadsheet.js"),
  "utf8",
);

function loadSpreadsheet(application) {
  const alerts = [];
  const context = {
    window: {
      alert: (message) => alerts.push(message),
      Application: application,
    },
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  return { alerts, api: context.window.XstarsSpreadsheet };
}

test("selection serialization matches SelectionPayload for scalar and one-column Value2", () => {
  const application = {
    ActiveSheet: { Name: "Data" },
    Selection: {
      Address: () => "$A$1:$A$3",
      Areas: { Count: 1 },
      Rows: { Count: 3 },
      Columns: { Count: 1 },
      Value2: ["Group", 1, undefined],
    },
  };
  const { api } = loadSpreadsheet(application);

  assert.deepEqual(
    JSON.parse(JSON.stringify(api.readSelection(application))),
    {
      version: "1.0",
      values: [["Group"], [1], [null]],
      address: "$A$1:$A$3",
      sheet: "Data",
    },
  );
  assert.deepEqual(
    JSON.parse(JSON.stringify(api.normalizeSelectionValues(5, 1, 1))),
    [[5]],
  );
});

test("Type=8 range prompts serialize the returned Range and cancellation is recoverable", () => {
  const range = {
    Address: "$C$1:$D$2",
    Rows: { Count: 2 },
    Columns: { Count: 2 },
    Value2: [["A", "B"], [1, 2]],
    Worksheet: { Name: "ELISA" },
  };
  const types = [];
  let next = range;
  const application = {
    ActiveSheet: { Name: "Data" },
    InputBox: (...args) => {
      types.push(args[7]);
      const result = next;
      next = false;
      return result;
    },
  };
  const { api } = loadSpreadsheet(application);

  const selected = api.promptRange(application, "message", "title");
  const cancelled = api.promptRange(application, "message", "title");

  assert.deepEqual(JSON.parse(JSON.stringify(selected)), {
    version: "1.0",
    values: [["A", "B"], [1, 2]],
    address: "$C$1:$D$2",
    sheet: "ELISA",
  });
  assert.equal(cancelled, null);
  assert.deepEqual(types, [8, 8]);
});

test("selection rejects discontiguous areas and non-finite values", () => {
  const application = {
    ActiveSheet: { Name: "Data" },
    Selection: {
      Address: "$A$1,$C$1",
      Areas: { Count: 2 },
      Rows: { Count: 1 },
      Columns: { Count: 2 },
      Value2: [1, 2],
    },
  };
  const { api } = loadSpreadsheet(application);

  assert.throws(() => api.readSelection(application), /单个连续选区/);
  assert.throws(
    () => api.normalizeSelectionValues([[Number.NaN]], 1, 1),
    /无效数字/,
  );
});

test("WritebackPlan writes resized ranges and inserts a named picture at its anchor", () => {
  const rangeCalls = [];
  const resizeCalls = [];
  const pictureCalls = [];
  const targets = [];
  const anchors = { D6: { Left: 420, Top: 160 } };
  const sheet = {
    Name: "Data",
    Range: (address) => {
      rangeCalls.push(address);
      if (anchors[address]) {
        return anchors[address];
      }
      return {
        Resize: (rows, columns) => {
          const target = { Value2: null };
          resizeCalls.push([address, rows, columns]);
          targets.push(target);
          return target;
        },
      };
    },
    Shapes: {
      AddPicture: (...args) => {
        pictureCalls.push(args);
        return { Name: "Picture 1" };
      },
    },
  };
  const application = { ActiveSheet: sheet, StatusBar: false };
  const { api } = loadSpreadsheet(application);
  const plan = {
    version: "1.0",
    tables: [
      { startCell: "D2", values: [["mean", "sd"], [2.5, 0.2]] },
    ],
    images: [
      {
        anchorCell: "D6",
        name: "XSTARS_Plot_1",
        pictureId: "XSTARS_20260831_abcdef123456",
        artifact: { path: "C:\\Temp\\xstars\\chart.png" },
        width: 320,
        height: 180,
      },
    ],
    statusMessage: "XSTARS: Quick Run complete",
  };

  const result = api.executeWritebackPlan(application, plan);

  assert.deepEqual(rangeCalls, ["D2", "D6"]);
  assert.deepEqual(resizeCalls, [["D2", 2, 2]]);
  assert.deepEqual(targets[0].Value2, plan.tables[0].values);
  assert.deepEqual(pictureCalls, [
    ["C:\\Temp\\xstars\\chart.png", 0, -1, 420, 160, 320, 180],
  ]);
  assert.equal(result.pictures[0].Name, "XSTARS_20260831_abcdef123456");
  assert.equal(application.StatusBar, "XSTARS: Quick Run complete");
});

test("WritebackPlan rejects a changed ActiveSheet before any write", () => {
  const writes = [];
  const application = {
    ActiveSheet: {
      Name: "Other",
      Range: (...args) => {
        writes.push(args);
        return {};
      },
    },
    StatusBar: false,
  };
  const { api } = loadSpreadsheet(application);

  assert.throws(
    () => api.executeWritebackPlan(
      application,
      {
        version: "1.0",
        tables: [{ startCell: "A1", values: [[1]] }],
        images: [],
      },
      { sheet: "Data" },
    ),
    /请切回“Data”.*未写入任何内容/,
  );
  assert.deepEqual(writes, []);
});

test("image writeback uses native dimensions when width and height are omitted", () => {
  const calls = [];
  const application = {
    ActiveSheet: {
      Name: "Data",
      Range: () => ({ Left: 25, Top: 50 }),
      Shapes: {
        AddPicture: (...args) => {
          calls.push(args);
          return {};
        },
      },
    },
  };
  const { api } = loadSpreadsheet(application);

  api.executeWritebackPlan(application, {
    version: "1.0",
    tables: [],
    images: [
      {
        anchorCell: "C3",
        name: "Plot",
        artifact: { path: "C:\\Temp\\plot.png" },
      },
    ],
    statusMessage: "done",
  });

  assert.deepEqual(calls[0], ["C:\\Temp\\plot.png", 0, -1, 25, 50, -1, -1]);
});

test("malformed plans fail before any host write and errors use status plus alert", () => {
  const writes = [];
  const application = {
    ActiveSheet: {
      Name: "Data",
      Range: (...args) => {
        writes.push(args);
        return {};
      },
    },
  };
  const { alerts, api } = loadSpreadsheet(application);

  assert.throws(
    () => api.executeWritebackPlan(application, {
      version: "1.0",
      tables: [{ startCell: "A1", values: [[1], [2, 3]] }],
      images: [],
    }),
    /不是矩形/,
  );
  assert.deepEqual(writes, []);
  api.showError(application, "选区无效");
  assert.equal(application.StatusBar, "XSTARS: 失败 — 选区无效");
  assert.deepEqual(alerts, ["选区无效"]);
});
