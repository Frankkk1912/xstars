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
  assert.deepEqual(
    JSON.parse(JSON.stringify(context.window.XstarsWpsAddin.CONTROL_COMMANDS)),
    { xstarsRun: "run", xstarsQuickRun: "run_quick" },
  );
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
  assert.deepEqual(busy.alerts, ["XSTARS 正在处理另一个任务，请稍后重试。"]);
  assert.match(busyHost.application.StatusBar, /正在处理另一个任务/);
});
