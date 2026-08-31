const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const { injectConfig } = require("../scripts/inject-config.cjs");
const template = path.join(__dirname, "..", "config.template.js");

test("development injection reads the broker token and generates executable config", (t) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "xstars-config-test-"));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const configPath = path.join(directory, "wps_service.json");
  const outputPath = path.join(directory, "config.js");
  const token = "secure-token-".padEnd(43, "x");
  fs.writeFileSync(configPath, JSON.stringify({ version: "1.0", token }));

  injectConfig({
    config: configPath,
    template,
    output: outputPath,
    port: "3988",
  });

  const generated = fs.readFileSync(outputPath, "utf8");
  assert.doesNotMatch(generated, /<port>|<token>/);
  const context = { window: {} };
  vm.createContext(context);
  vm.runInContext(generated, context);
  assert.equal(context.window.XSTARS_WPS_CONFIG.port, 3988);
  assert.equal(context.window.XSTARS_WPS_CONFIG.token, token);
});

test("port precedence is --port then env then persisted file then default", (t) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "xstars-config-test-"));
  const previous = process.env.XSTARS_WPS_PORT;
  t.after(() => {
    if (previous === undefined) {
      delete process.env.XSTARS_WPS_PORT;
    } else {
      process.env.XSTARS_WPS_PORT = previous;
    }
    fs.rmSync(directory, { recursive: true, force: true });
  });
  const configPath = path.join(directory, "wps_service.json");
  const token = "secure-token-".padEnd(43, "x");
  const readPort = (outputPath) => {
    const context = { window: {} };
    vm.createContext(context);
    vm.runInContext(fs.readFileSync(outputPath, "utf8"), context);
    return context.window.XSTARS_WPS_CONFIG.port;
  };

  fs.writeFileSync(configPath, JSON.stringify({ version: "1.0", token, port: 4011 }));
  delete process.env.XSTARS_WPS_PORT;
  const fromFile = path.join(directory, "from-file.js");
  injectConfig({ config: configPath, template, output: fromFile });
  assert.equal(readPort(fromFile), 4011);

  process.env.XSTARS_WPS_PORT = "4012";
  const fromEnv = path.join(directory, "from-env.js");
  injectConfig({ config: configPath, template, output: fromEnv });
  assert.equal(readPort(fromEnv), 4012);

  const explicit = path.join(directory, "explicit.js");
  injectConfig({ config: configPath, template, output: explicit, port: "4013" });
  assert.equal(readPort(explicit), 4013);

  fs.writeFileSync(configPath, JSON.stringify({ version: "1.0", token }));
  delete process.env.XSTARS_WPS_PORT;
  const fallback = path.join(directory, "fallback.js");
  injectConfig({ config: configPath, template, output: fallback });
  assert.equal(readPort(fallback), 3892);
});

test("development injection rejects a missing or short token without output", (t) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "xstars-config-test-"));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const configPath = path.join(directory, "wps_service.json");
  const outputPath = path.join(directory, "config.js");
  fs.writeFileSync(configPath, JSON.stringify({ version: "1.0", token: "short" }));

  assert.throws(
    () => injectConfig({ config: configPath, template, output: outputPath }),
    /valid token/,
  );
  assert.equal(fs.existsSync(outputPath), false);
});
