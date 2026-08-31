const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index];
    const value = argv[index + 1];
    if (!flag || !flag.startsWith("--") || value === undefined) {
      throw new Error("arguments must be --name value pairs");
    }
    args[flag.slice(2)] = value;
  }
  return args;
}

function injectConfig(options) {
  const addonRoot = path.resolve(__dirname, "..");
  const templatePath = path.resolve(
    options.template || path.join(addonRoot, "config.template.js"),
  );
  const serviceConfigPath = path.resolve(
    options.config || path.join(os.homedir(), ".xstars", "wps_service.json"),
  );
  const outputPath = path.resolve(options.output || path.join(addonRoot, "config.js"));
  const port = Number(options.port || process.env.XSTARS_WPS_PORT || 3892);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error("port must be an integer from 1 to 65535");
  }

  const config = JSON.parse(fs.readFileSync(serviceConfigPath, "utf8"));
  if (!config || typeof config.token !== "string" || config.token.length < 32) {
    throw new Error("service config does not contain a valid token");
  }
  const template = fs.readFileSync(templatePath, "utf8");
  if (!template.includes("<port>") || !template.includes('"<token>"')) {
    throw new Error("config template placeholders are missing");
  }
  const generated = template
    .replaceAll("<port>", String(port))
    .replaceAll('"<token>"', JSON.stringify(config.token));
  if (generated.includes("<port>") || generated.includes("<token>")) {
    throw new Error("config template contains unresolved placeholders");
  }

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const temporary = `${outputPath}.${process.pid}.tmp`;
  try {
    fs.writeFileSync(temporary, generated, { encoding: "utf8", mode: 0o600 });
    fs.renameSync(temporary, outputPath);
    fs.chmodSync(outputPath, 0o600);
  } finally {
    fs.rmSync(temporary, { force: true });
  }
  return outputPath;
}

if (require.main === module) {
  try {
    injectConfig(parseArgs(process.argv.slice(2)));
  } catch (error) {
    process.stderr.write(`XSTARS config injection failed: ${error.message}\n`);
    process.exitCode = 1;
  }
}

module.exports = { injectConfig, parseArgs };
