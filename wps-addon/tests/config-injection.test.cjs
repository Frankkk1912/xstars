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
