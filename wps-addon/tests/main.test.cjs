const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const sources = ["service-client.js", "spreadsheet.js", "main.js"].map((file) =>
  fs.readFileSync(path.join(__dirname, "..", file), "utf8"),
);

function jsonResponse(body, ok = true, status = 200) {
  return {
    ok,
    status,
    text: async () => JSON.stringify(body),
  };
}

function loadAddin(fetchHandler, application) {
  const alerts = [];
  const fetchCalls = [];
  const context = {
    AbortController,
    setTimeout,
    window: {
      Application: application,
      XSTARS_WPS_CONFIG: {
        port: 3892,
        token: "m".repeat(43),
        healthRetries: 1,
      },
      alert: (message) => alerts.push(message),
      fetch: async (url, init) => {
        fetchCalls.push({ url, init });
        return await fetchHandler(url, init);
      },
    },
  };
  vm.createContext(context);
  for (const source of sources) {
    vm.runInContext(source, context);
  }
  return { alerts, context, fetchCalls };
}

function makeHost() {
  const tableWrites = [];
  const pictures = [];
  const sheet = {
    Name: "Data",
    Range: (address) => {
      if (address === "D5") {
        return { Left: 400, Top: 200 };
      }
      return {
        Resize: (rows, columns) => {
          const target = {
            set Value2(value) {
              tableWrites.push({ address, rows, columns, value });
            },
          };
          return target;
        },
      };
    },
    Shapes: {
      AddPicture: (...args) => {
        const picture = { Name: "Picture 1" };
        pictures.push({ args, picture });
        return picture;
      },
    },
  };
  const application = {
    ActiveSheet: sheet,
    Selection: {
      Address: () => "$A$1:$B$3",
      Areas: { Count: 1 },
      Rows: { Count: 3 },
      Columns: { Count: 2 },
      Value2: [
        ["Control", "Treatment"],
        [1, 2],
        [3, 4],
      ],
    },
    StatusBar: false,
  };
  return { application, pictures, tableWrites };
}

test("Run and Quick Run serialize selection, call the broker, and execute WritebackPlan", async () => {
  const commands = [];
  const host = makeHost();
  const { context, fetchCalls } = loadAddin((url, init) => {
    if (url.endsWith("/health")) {
      return jsonResponse({
        ok: true,
        service: "xstars-wps-service",
        port: 3892,
      });
    }
    const request = JSON.parse(init.body);
    commands.push(request.command);
    return jsonResponse({
      version: "1.0",
      ok: true,
      command: request.command,
      writebackPlan: {
        version: "1.0",
        tables: [{ startCell: "D2", values: [["mean"], [2.5]] }],
        images: [{
          anchorCell: "D5",
          name: `XSTARS_${request.command}`,
          artifact: { path: "C:\\Temp\\chart.png" },
          width: 300,
          height: 180,
        }],
        statusMessage: `XSTARS: ${request.command} complete`,
      },
    });
  }, host.application);

  const runResult = await context.window.XstarsWpsAddin.runCommand("run");
  const quickResult = await context.window.XstarsWpsAddin.runCommand("run_quick");

  assert.equal(runResult.selection.sheet, "Data");
  assert.equal(quickResult.selection.address, "$A$1:$B$3");
  assert.deepEqual(commands, ["run", "run_quick"]);
  assert.deepEqual(host.tableWrites.map((item) => item.address), ["D2", "D2"]);
  assert.equal(host.pictures[0].picture.Name, "XSTARS_run");
  assert.equal(host.pictures[1].picture.Name, "XSTARS_run_quick");
  assert.equal(host.application.StatusBar, "XSTARS: run_quick complete");
  assert.equal(fetchCalls.filter((call) => call.url.endsWith("/health")).length, 1);
  const commandRequest = JSON.parse(
    fetchCalls.find((call) => call.url.endsWith("/command")).init.body,
  );
  assert.deepEqual(commandRequest.selection, {
    version: "1.0",
    values: [
      ["Control", "Treatment"],
      [1, 2],
      [3, 4],
    ],
    address: "$A$1:$B$3",
    sheet: "Data",
  });
});

test("an async command refuses writeback after ActiveSheet changes", async () => {
  const host = makeHost();
  let resolveCommand;
  let commandStarted;
  const started = new Promise((resolve) => {
    commandStarted = resolve;
  });
  const pending = new Promise((resolve) => {
    resolveCommand = resolve;
  });
  const { context, alerts } = loadAddin((url) => {
    if (url.endsWith("/health")) {
      return jsonResponse({ ok: true, service: "xstars-wps-service", port: 3892 });
    }
    commandStarted();
    return pending;
  }, host.application);

  const request = context.window.XstarsWpsAddin.runCommand("run_quick");
  await started;
  const wrongSheetWrites = [];
  host.application.ActiveSheet = {
    Name: "Other",
    Range: (...args) => {
      wrongSheetWrites.push(args);
      return {};
    },
  };
  resolveCommand(jsonResponse({
    ok: true,
    writebackPlan: {
      version: "1.0",
      tables: [{ startCell: "D2", values: [[1]] }],
      images: [],
      statusMessage: "done",
    },
  }));

  const result = await request;

  assert.match(result.message, /请切回“Data”/);
  assert.deepEqual(wrongSheetWrites, []);
  assert.equal(host.tableWrites.length, 0);
  assert.equal(alerts.length, 1);
});

test("OnAction dispatches the approved Quick Run control asynchronously", async () => {
  const host = makeHost();
  let resolveCommand;
  const commandSeen = new Promise((resolve) => {
    resolveCommand = resolve;
  });
  const { context } = loadAddin((url, init) => {
    if (url.endsWith("/health")) {
      return jsonResponse({
        ok: true,
        service: "xstars-wps-service",
        port: 3892,
      });
    }
    resolveCommand(JSON.parse(init.body).command);
    return jsonResponse({
      version: "1.0",
      ok: true,
      writebackPlan: {
        version: "1.0",
        tables: [],
        images: [],
        statusMessage: "XSTARS: Quick Run complete",
      },
    });
  }, host.application);

  assert.equal(context.window.OnAction({ Id: "xstarsQuickRun" }), true);
  assert.equal(await commandSeen, "run_quick");
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(host.application.StatusBar, "XSTARS: Quick Run complete");
});

test("Ribbon callbacks expose the approved controls and self-owned icon paths", () => {
  const host = makeHost();
  const { context } = loadAddin(
    () => jsonResponse({ ok: false }, false, 500),
    host.application,
  );
  const ribbon = { Invalidate: () => {} };

  assert.equal(context.window.OnAddinLoad(ribbon), true);
  assert.equal(host.application.ribbonUI, ribbon);
  assert.equal(context.window.OnAction({ Id: "unrelated" }), true);
  assert.equal(context.window.GetImage({ Id: "xstarsRun" }), "assets/run.svg");
  assert.equal(
    context.window.GetImage({ Id: "xstarsQuickRun" }),
    "assets/quick-run.svg",
  );
  const controls = JSON.parse(
    JSON.stringify(context.window.XstarsWpsAddin.CONTROL_COMMANDS),
  );
  assert.equal(controls.xstarsRun, "run");
  assert.equal(controls.xstarsQuickRun, "run_quick");
  assert.equal(controls.xstarsWB, "run_wb");
  assert.equal(controls.xstarsQPCR, "run_qpcr");
  assert.equal(controls.xstarsCCK8, "run_cck8");
  assert.equal(controls.xstarsELISA, "run_elisa");
  assert.equal(controls.xstarsTransform, "run_transform_only");
  assert.equal(controls.xstarsStandardCurve, "run_standard_curve");
  assert.equal(controls.xstarsExport, "run_export");
  assert.equal(controls.xstarsThemeNature, "run_set_theme_nature");
  assert.equal(
    controls.xstarsJournalPaletteNature,
    "run_set_journal_palette_nature",
  );
  assert.equal(controls.xstarsPaletteColorblind, "run_set_palette_colorblind");
  assert.equal(controls.xstarsResetSettings, "run_reset_settings");
});

test("ELISA uses two Type=8 InputBox ranges and sends both selections", async () => {
  const host = makeHost();
  const ranges = [
    {
      Address: () => "$A$1:$C$4",
      Rows: { Count: 4 },
      Columns: { Count: 3 },
      Value2: [[1, 10, 100], [0.1, 1, 10], [0.11, 1.1, 10.1], [0.09, 0.9, 9.9]],
      Worksheet: host.application.ActiveSheet,
    },
    {
      Address: () => "$E$1:$F$4",
      Rows: { Count: 4 },
      Columns: { Count: 2 },
      Value2: [["Control", "Treatment"], [0.2, 0.4], [0.21, 0.42], [0.19, 0.38]],
      Worksheet: host.application.ActiveSheet,
    },
  ];
  const inputTypes = [];
  host.application.InputBox = (...args) => {
    inputTypes.push(args[7]);
    return ranges.shift();
  };
  let commandBody;
  const { context } = loadAddin((url, init) => {
    if (url.endsWith("/health")) {
      return jsonResponse({ ok: true, service: "xstars-wps-service", port: 3892 });
    }
    commandBody = JSON.parse(init.body);
    return jsonResponse({
      ok: true,
      writebackPlan: { version: "1.0", tables: [], images: [], statusMessage: "done" },
    });
  }, host.application);

  const result = await context.window.XstarsWpsAddin.runElisa();

  assert.equal(result.response.ok, true);
  assert.deepEqual(inputTypes, [8, 8]);
  assert.equal(commandBody.command, "run_elisa");
  assert.equal(commandBody.selection.address, "$A$1:$C$4");
  assert.equal(commandBody.sampleSelection.address, "$E$1:$F$4");
});

test("Standard Curve uses two Type=8 selections and sends samples for optional back-calculation", async () => {
  const host = makeHost();
  const ranges = [
    {
      Address: () => "$A$1:$C$3",
      Rows: { Count: 3 },
      Columns: { Count: 3 },
      Value2: [[1, 10, 100], [0.1, 1, 10], [0.11, 1.1, 10.1]],
      Worksheet: host.application.ActiveSheet,
    },
    {
      Address: () => "$E$1:$F$3",
      Rows: { Count: 3 },
      Columns: { Count: 2 },
      Value2: [["Control", "Treatment"], [0.2, 0.4], [0.3, 0.5]],
      Worksheet: host.application.ActiveSheet,
    },
  ];
  const inputTypes = [];
  host.application.InputBox = (...args) => {
    inputTypes.push(args[7]);
    return ranges.shift();
  };
  const commandBodies = [];
  const { context } = loadAddin((url, init) => {
    if (url.endsWith("/health")) {
      return jsonResponse({ ok: true, service: "xstars-wps-service", port: 3892 });
    }
    const body = JSON.parse(init.body);
    commandBodies.push(body);
    if (body.stage === "configure") {
      return jsonResponse({
        ok: true,
        continuation: { fitMethod: "linear", backCalculate: true },
        writebackPlan: { version: "1.0", tables: [], images: [], statusMessage: "configured" },
      });
    }
    return jsonResponse({
      ok: true,
      writebackPlan: { version: "1.0", tables: [], images: [], statusMessage: "done" },
    });
  }, host.application);

  const result = await context.window.XstarsWpsAddin.runStandardCurve();

  assert.equal(result.response.ok, true);
  assert.deepEqual(inputTypes, [8, 8]);
  assert.deepEqual(commandBodies.map((body) => body.stage), ["configure", "execute"]);
  assert.equal(commandBodies[1].command, "run_standard_curve");
  assert.equal(commandBodies[1].selection.address, "$A$1:$C$3");
  assert.equal(commandBodies[1].sampleSelection.address, "$E$1:$F$3");
  assert.deepEqual(commandBodies[1].curveOptions, {
    fitMethod: "linear",
    backCalculate: true,
  });
});

test("Standard Curve sample-selection cancellation sends no execute-stage request", async () => {
  const host = makeHost();
  let prompts = 0;
  host.application.InputBox = () => {
    prompts += 1;
    return prompts === 1
      ? {
          Address: () => "$A$1:$C$3",
          Rows: { Count: 3 },
          Columns: { Count: 3 },
          Value2: [[1, 10, 100], [0.1, 1, 10], [0.11, 1.1, 10.1]],
          Worksheet: host.application.ActiveSheet,
        }
      : false;
  };
  const stages = [];
  const { context } = loadAddin((url, init) => {
    if (url.endsWith("/health")) {
      return jsonResponse({ ok: true, service: "xstars-wps-service", port: 3892 });
    }
    const body = JSON.parse(init.body);
    stages.push(body.stage);
    return jsonResponse({
      ok: true,
      continuation: { fitMethod: "linear", backCalculate: true },
      writebackPlan: { version: "1.0", tables: [], images: [], statusMessage: "configured" },
    });
  }, host.application);

  const result = await context.window.XstarsWpsAddin.runStandardCurve();

  assert.deepEqual(JSON.parse(JSON.stringify(result)), {
    cancelled: true,
    stage: "sample",
  });
  assert.deepEqual(stages, ["configure"]);
});

test("Standard Curve skips the second InputBox when back-calculation is disabled", async () => {
  const host = makeHost();
  let prompts = 0;
  host.application.InputBox = () => {
    prompts += 1;
    return {
      Address: () => "$A$1:$C$3",
      Rows: { Count: 3 },
      Columns: { Count: 3 },
      Value2: [[1, 10, 100], [0.1, 1, 10], [0.11, 1.1, 10.1]],
      Worksheet: host.application.ActiveSheet,
    };
  };
  const bodies = [];
  const { context } = loadAddin((url, init) => {
    if (url.endsWith("/health")) {
      return jsonResponse({ ok: true, service: "xstars-wps-service", port: 3892 });
    }
    const body = JSON.parse(init.body);
    bodies.push(body);
    return body.stage === "configure"
      ? jsonResponse({
          ok: true,
          continuation: { fitMethod: "linear", backCalculate: false },
          writebackPlan: { version: "1.0", tables: [], images: [], statusMessage: "configured" },
        })
      : jsonResponse({
          ok: true,
          writebackPlan: { version: "1.0", tables: [], images: [], statusMessage: "done" },
        });
  }, host.application);

  const result = await context.window.XstarsWpsAddin.runStandardCurve();

  assert.equal(result.response.ok, true);
  assert.equal(prompts, 1);
  assert.equal(bodies[1].sampleSelection, undefined);
  assert.equal(bodies[1].curveOptions.backCalculate, false);
});

test("ELISA cancellation is friendly and sends no service request", async () => {
  const host = makeHost();
  host.application.InputBox = () => false;
  let requests = 0;
  const { context, alerts } = loadAddin(() => {
    requests += 1;
    throw new Error("network must not be called");
  }, host.application);

  const result = await context.window.XstarsWpsAddin.runElisa();

  assert.deepEqual(JSON.parse(JSON.stringify(result)), {
    cancelled: true,
    stage: "standard",
  });
  assert.equal(requests, 0);
  assert.deepEqual(alerts, []);
  assert.equal(host.application.StatusBar, "XSTARS: ELISA 已取消");
});

test("high-resolution export prefers pictureId and arbitrary Shapes use CopyPicture", async () => {
  const host = makeHost();
  let currentShape = { Name: "XSTARS_20260831_abcdef123456" };
  host.application.Selection = {
    ShapeRange: { Item: () => currentShape },
  };
  const prompts = ["png", "300", "tiff", "96"];
  host.application.InputBox = (...args) => {
    assert.equal(args[7], 2);
    return prompts.shift();
  };
  const bodies = [];
  const { context } = loadAddin((url, init) => {
    if (url.endsWith("/health")) {
      return jsonResponse({ ok: true, service: "xstars-wps-service", port: 3892 });
    }
    const body = JSON.parse(init.body);
    bodies.push(body);
    return jsonResponse({
      ok: true,
      export: { path: "C:\\Users\\test\\.xstars\\exports\\chart.png" },
      writebackPlan: { version: "1.0", tables: [], images: [], statusMessage: "exported" },
    });
  }, host.application);

  await context.window.XstarsWpsAddin.runExport();
  let copied = 0;
  currentShape = {
    Name: "Picture 1",
    CopyPicture: (appearance, format) => {
      copied += 1;
      assert.deepEqual([appearance, format], [2, -4147]);
    },
  };
  await context.window.XstarsWpsAddin.runExport();

  assert.equal(bodies[0].export.pictureId, "XSTARS_20260831_abcdef123456");
  assert.equal(bodies[0].export.clipboard, false);
  assert.equal(bodies[1].export.clipboard, true);
  assert.equal(bodies[1].export.format, "tiff");
  assert.equal(copied, 1);
});

test("selection and broker errors are shown without triggering real dialogs", async () => {
  const host = makeHost();
  host.application.Selection.Areas.Count = 2;
  const unavailable = loadAddin(
    () => {
      throw new Error("network must not be called");
    },
    host.application,
  );

  const selectionResult = await unavailable.context.window.XstarsWpsAddin.runCommand(
    "run_quick",
  );

  assert.match(selectionResult.message, /单个连续选区/);
  assert.deepEqual(unavailable.alerts, ["仅支持单个连续选区"]);

  const busyHost = makeHost();
  const busy = loadAddin((url) => {
    if (url.endsWith("/health")) {
      return jsonResponse({
        ok: true,
        service: "xstars-wps-service",
        port: 3892,
      });
    }
    return jsonResponse(
      { ok: false, error: { code: "BUSY", message: "busy" } },
      false,
      409,
    );
  }, busyHost.application);

  const busyResult = await busy.context.window.XstarsWpsAddin.runCommand("run");

  assert.equal(busyResult.error.code, "BUSY");
  assert.deepEqual(busy.alerts, [
    "XSTARS 正在处理另一个任务，请稍后重试。\n详情：busy",
  ]);
  assert.match(busyHost.application.StatusBar, /正在处理另一个任务/);
});
