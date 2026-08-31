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

test("cancel aborts an active browser request", async () => {
  const api = loadClient((url, init) => {
    if (url.endsWith("/health")) {
      return Promise.resolve(jsonResponse({
        ok: true,
        service: "xstars-wps-service",
        port: 3892,
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
  assert.equal(client.cancelActiveRequest(), false);
});
