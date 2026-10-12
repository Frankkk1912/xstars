const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "service-client.js"),
  "utf8",
);

function jsonResponse(body, ok = true, status = 200) {
  return {
    ok,
    status,
    text: async () => JSON.stringify(body),
  };
}

function loadClient(fetchImpl) {
  const context = {
    AbortController,
    setTimeout,
    window: { fetch: fetchImpl },
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  return context.window.XstarsServiceClient;
}

const token = "t".repeat(43);

test("service discovery retries injected ports until the authenticated broker is healthy", async () => {
  const calls = [];
  let firstPortFailures = 0;
  const api = loadClient(async (url) => {
    calls.push(url);
    if (url.includes(":3892/")) {
      firstPortFailures += 1;
      throw new Error("connection refused");
    }
    return jsonResponse({
      version: "1.0",
      ok: true,
      service: "xstars-wps-service",
      port: 3893,
    });
  });
  const sleeps = [];
  const client = new api.WpsServiceClient(
    { token, ports: [3892, 3893], healthRetries: 2, retryIntervalMs: 5 },
    { sleep: async (ms) => sleeps.push(ms) },
  );

  const health = await client.discoverService();

  assert.equal(health.port, 3893);
  assert.equal(client.baseUrl, "http://127.0.0.1:3893");
  assert.equal(firstPortFailures, 1);
  assert.deepEqual(sleeps, []);
  assert.deepEqual(calls, [
    "http://127.0.0.1:3892/health",
    "http://127.0.0.1:3893/health",
  ]);
});

test("health check retries after every configured port is unavailable", async () => {
  let calls = 0;
  const api = loadClient(async () => {
    calls += 1;
    if (calls < 2) {
      throw new Error("not ready");
    }
    return jsonResponse({
      ok: true,
      service: "xstars-wps-service",
      port: 3892,
    });
  });
  const sleeps = [];
  const client = new api.WpsServiceClient(
    { token, port: 3892, healthRetries: 2, retryIntervalMs: 7 },
    { sleep: async (ms) => sleeps.push(ms) },
  );

  await client.discoverService();

  assert.equal(calls, 2);
  assert.deepEqual(sleeps, [7]);
});

test("command sends the schema DTO and bearer token", async () => {
  const calls = [];
  const api = loadClient(async (url, init) => {
    calls.push({ url, init });
    if (url.endsWith("/health")) {
      return jsonResponse({
        ok: true,
        service: "xstars-wps-service",
        port: 3892,
      });
    }
    return jsonResponse({
      version: "1.0",
      ok: true,
      writebackPlan: {
        version: "1.0",
        tables: [],
        images: [],
        statusMessage: "done",
      },
    });
  });
  const client = new api.WpsServiceClient({ token, port: 3892 });
  const selection = {
    version: "1.0",
    values: [["A"], [1]],
    address: "$A$1:$A$2",
    sheet: "Data",
  };

  await client.command("run_quick", selection, {});

  assert.equal(calls[1].url, "http://127.0.0.1:3892/command");
  assert.equal(calls[1].init.headers.Authorization, `Bearer ${token}`);
  assert.equal(calls[1].init.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(calls[1].init.body), {
    version: "1.0",
    command: "run_quick",
    selection,
    config: {},
  });
});

test("ELISA and export command extras are explicitly serialized", async () => {
  const bodies = [];
  const api = loadClient(async (url, init) => {
    if (url.endsWith("/health")) {
      return jsonResponse({ ok: true, service: "xstars-wps-service", port: 3892 });
    }
    bodies.push(JSON.parse(init.body));
    return jsonResponse({
      ok: true,
      writebackPlan: { version: "1.0", tables: [], images: [], statusMessage: "done" },
    });
  });
  const client = new api.WpsServiceClient({ token, port: 3892 });
  const standard = { version: "1.0", values: [[1], [0.1]], address: "A1:A2", sheet: "ELISA" };
  const sample = { version: "1.0", values: [["A"], [0.2]], address: "C1:C2", sheet: "ELISA" };

  await client.command("run_elisa", standard, {}, { sampleSelection: sample });
  await client.command("run_export", null, {}, {
    export: { pictureId: "XSTARS_20260831_abcdef123456", format: "png", dpi: 300 },
  });

  assert.deepEqual(bodies[0].sampleSelection, sample);
  assert.equal(bodies[1].selection, undefined);
  assert.equal(bodies[1].export.dpi, 300);
});

test("known service errors map to stable user-facing messages", async () => {
  const api = loadClient(async (url) => {
    if (url.endsWith("/health")) {
      return jsonResponse({
        ok: true,
        service: "xstars-wps-service",
        port: 3892,
      });
    }
    return jsonResponse(
      { ok: false, error: { code: "BUSY", message: "another job" } },
      false,
      409,
    );
  });
  const client = new api.WpsServiceClient({ token, port: 3892 });

  await assert.rejects(
    client.command("run", { version: "1.0" }, {}),
    (error) => {
      assert.equal(error.code, "BUSY");
      assert.equal(error.status, 409);
      assert.equal(
        api.toUserMessage(error),
        "XSTARS 正在处理另一个任务，请稍后重试。\n详情：another job",
      );
      return true;
    },
  );
});

test("analysis errors append bounded server detail to the stable message", () => {
  const api = loadClient(async () => jsonResponse({}));
  const elisaDetail = "Column name 'Instructions' cannot be parsed as a concentration.";
  const message = api.toUserMessage(
    new api.ServiceError("ANALYSIS_FAILED", elisaDetail, 422),
  );

  assert.equal(
    message,
    `XSTARS 无法分析当前数据，请检查数据格式。\n详情：${elisaDetail}`,
  );

  const longMessage = api.toUserMessage(
    new api.ServiceError("ANALYSIS_FAILED", "x".repeat(250), 422),
  );
  assert.equal(longMessage.split("详情：")[1].length, 200);
});

test("missing generated config fails diagnostically without a hard-coded token", () => {
  const api = loadClient(async () => jsonResponse({}));

  assert.throws(
    () => new api.WpsServiceClient({ port: 3892, token: "<token>" }),
    (error) => {
      assert.equal(error.code, "CONFIG_MISSING");
      assert.match(api.toUserMessage(error), /配置未生成/);
      return true;
    },
  );
});

test("cancel aborts an active browser request and asks the broker to stop its worker", async () => {
  const calls = [];
  const api = loadClient((url, init) => {
    calls.push(url);
    if (url.endsWith("/health")) {
      return Promise.resolve(jsonResponse({
        ok: true,
        service: "xstars-wps-service",
        port: 3892,
      }));
    }
    if (url.endsWith("/cancel")) {
      assert.equal(init.headers.Authorization, `Bearer ${token}`);
      return Promise.resolve(jsonResponse({
        ok: true,
        cancelled: true,
        status: "cancelled",
        busy: false,
      }));
    }
    return new Promise((_resolve, reject) => {
      init.signal.addEventListener("abort", () => {
        const error = new Error("aborted");
        error.name = "AbortError";
        reject(error);
      });
    });
  });
  const client = new api.WpsServiceClient({ token, port: 3892 });
  await client.discoverService();
  const request = client.command("run_quick", { version: "1.0" }, {});

  assert.equal(client.cancelActiveRequest(), true);
  await assert.rejects(request, (error) => error.code === "CANCELLED");
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(calls.filter((url) => url.endsWith("/cancel")).length, 1);
  assert.equal(client.cancelActiveRequest(), false);
});

function abortableWait(signal) {
  return new Promise((_resolve, reject) => {
    const fail = () => { const error = new Error("aborted"); error.name = "AbortError"; reject(error); };
    if (signal.aborted) { fail(); } else { signal.addEventListener("abort", fail, { once: true }); }
  });
}

test("dialog command and interaction GET run concurrently, then POST supplies selection", async () => {
  let finishCommand;
  const calls = [];
  const selected = { version: "1.0", values: [["A", "B"], [1, 2]], address: "$A$1:$B$2", sheet: "Data" };
  let first = true;
  const api = loadClient(async (url, init) => {
    calls.push({ url, init });
    if (url.endsWith("/command")) {
      return new Promise((resolve) => { finishCommand = resolve; });
    }
    if (init.method === "POST") {
      assert.deepEqual(JSON.parse(init.body), { id: "i".repeat(32), selection: selected });
      finishCommand(jsonResponse({ ok: true, writebackPlan: { tables: [], images: [] } }));
      return jsonResponse({ ok: true });
    }
    if (first) {
      first = false;
      assert.equal(typeof finishCommand, "function");
      return jsonResponse({ ok: true, interaction: { id: "i".repeat(32), prompt: "Choose" } });
    }
    return abortableWait(init.signal);
  });
  const client = new api.WpsServiceClient({ token, port: 3892 });
  client.baseUrl = "http://127.0.0.1:3892";
  let prompted = 0;
  await client.command("run", null, {}, { interactionEnabled: true }, {
    onInteraction: () => { prompted++; return selected; },
  });
  assert.equal(prompted, 1);
  assert.equal(calls[0].url.endsWith("/command"), true);
  assert.equal(calls[1].url.endsWith("/interaction"), true);
  assert.equal(calls[1].init.headers.Authorization, `Bearer ${token}`);
  assert.equal(calls[1].init.signal.aborted, true);
  assert.equal(client.activeController, null);
});

test("interaction 404 marks unavailable, cancels old job before fallback and aborts fetches", async () => {
  const calls = [];
  const api = loadClient(async (url, init) => {
    calls.push(url);
    if (url.endsWith("/command")) { return abortableWait(init.signal); }
    if (url.endsWith("/cancel")) { return jsonResponse({ ok: true, cancelled: true }); }
    return jsonResponse({ ok: false, error: { code: "NOT_FOUND" } }, false, 404);
  });
  const client = new api.WpsServiceClient({ token, port: 3892 });
  client.baseUrl = "http://127.0.0.1:3892";
  await assert.rejects(client.command("run", null, {}, {}, { onInteraction: () => null }),
    (error) => error.code === "INTERACTION_UNAVAILABLE");
  assert.equal(calls.filter((url) => url.endsWith("/cancel")).length, 1);
  assert.equal(client.activeController, null);
});

test("cancelActiveRequest aborts command and outstanding interaction polling", async () => {
  const signals = [];
  const api = loadClient(async (url, init) => {
    if (url.endsWith("/cancel")) { return jsonResponse({ ok: true, cancelled: true }); }
    signals.push(init.signal);
    return abortableWait(init.signal);
  });
  const client = new api.WpsServiceClient({ token, port: 3892 });
  client.baseUrl = "http://127.0.0.1:3892";
  const command = client.command("run", null, {}, {}, { onInteraction: () => null });
  assert.equal(client.cancelActiveRequest(), true);
  await assert.rejects(command, (error) => error.code === "CANCELLED");
  assert.equal(signals.length, 2);
  assert.equal(signals.every((signal) => signal.aborted), true);
});

for (const failure of ["GET network", "GET JSON", "POST error", "POST JSON", "picker"]) {
  test(`any interaction failure cancels the active job and aborts fetches (${failure})`, async () => {
    const calls = [];
    const api = loadClient(async (url, init) => {
      calls.push({ url, init });
      if (url.endsWith("/command")) { return abortableWait(init.signal); }
      if (url.endsWith("/cancel")) { return jsonResponse({ ok: true, cancelled: true }); }
      if (init.method === "GET") {
        if (failure === "GET network") { throw new Error("offline"); }
        if (failure === "GET JSON") { return { ok: true, status: 200, text: async () => "invalid" }; }
        return jsonResponse({ ok: true, interaction: { id: "a".repeat(32), prompt: "Choose" } });
      }
      if (failure === "POST JSON") { return { ok: true, status: 200, text: async () => "invalid" }; }
      return jsonResponse({ ok: false, error: { code: "INTERNAL_ERROR" } }, false, 500);
    });
    const client = new api.WpsServiceClient({ token, port: 3892 });
    client.baseUrl = "http://127.0.0.1:3892";
    await assert.rejects(client.command("run", null, {}, {}, { onInteraction: () => {
      if (failure === "picker") { throw new Error("picker failed"); }
      return { workbook: "Book1.xlsx" };
    } }));
    assert.equal(calls.filter((call) => call.url.endsWith("/cancel")).length, 1);
    const activeCalls = calls.filter((call) => !call.url.endsWith("/cancel"));
    assert.equal(activeCalls.every((call) => call.init.signal.aborted), true);
    assert.equal(client.activeController, null);
  });
}

test("interaction errors have specific user-facing messages", () => {
  const api = loadClient(() => {});
  assert.match(api.toUserMessage(new api.ServiceError("INTERACTION_UNAVAILABLE", "")), /不支持弹窗内选择数据/);
  assert.match(api.toUserMessage(new api.ServiceError("NO_INTERACTION", "")), /选区请求已过期/);
});
