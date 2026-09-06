const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const projectRoot = path.resolve(__dirname, "..");
const project = require(path.join(projectRoot, "package.json"));
const buildRoot = path.join(projectRoot, "wps-addon-build");
const archivePath = path.join(buildRoot, `${project.name}.7z`);
const publishRoot = path.join(projectRoot, "wps-addon-publish");
const publishHtmlPath = path.join(publishRoot, "publish.html");
const deployRoot = path.join(projectRoot, "deploy");
const serverUrl =
    process.env.XSTARS_WPS_PUBLISH_BASE || "http://127.0.0.1:3890/addin/";

function assertLoopbackUrl(value) {
    let parsed;
    try {
        parsed = new URL(value);
    } catch (error) {
        throw new Error(`Invalid XSTARS_WPS_PUBLISH_BASE: ${error.message}`);
    }
    if (
        parsed.protocol !== "http:" ||
        parsed.hostname !== "127.0.0.1" ||
        !parsed.pathname.endsWith("/")
    ) {
        throw new Error(
            "XSTARS_WPS_PUBLISH_BASE must be an http://127.0.0.1 URL ending in '/'.",
        );
    }
}

function run(command, args) {
    const result = spawnSync(command, args, {
        cwd: projectRoot,
        encoding: "utf8",
        stdio: ["ignore", "pipe", "pipe"],
    });
    process.stdout.write(result.stdout || "");
    process.stderr.write(result.stderr || "");
    if (result.status !== 0) {
        throw new Error(`${command} exited with status ${result.status}`);
    }
}

async function waitForArchive() {
    let previousSize = -1;
    for (let attempt = 0; attempt < 200; attempt += 1) {
        if (fs.existsSync(archivePath)) {
            const currentSize = fs.statSync(archivePath).size;
            if (currentSize > 0 && currentSize === previousSize) {
                return;
            }
            previousSize = currentSize;
        }
        await new Promise((resolve) => setTimeout(resolve, 50));
    }
    throw new Error(`Timed out waiting for ${archivePath}`);
}

function readPublishList() {
    const html = fs.readFileSync(publishHtmlPath, "utf8");
    const match = html.match(/var curList = (\[[\s\S]*?\]);/);
    if (!match) {
        throw new Error("Could not locate the generated publish list.");
    }
    try {
        return JSON.parse(match[1]);
    } catch (error) {
        throw new Error(
            `The generated publish list is invalid JSON: ${error.message}`,
        );
    }
}

async function main() {
    assertLoopbackUrl(serverUrl);

    const wpsjsRoot = path.resolve(
        path.dirname(require.resolve("wpsjs/vite_plugins")),
        "..",
    );
    const buildModule = require(path.join(wpsjsRoot, "src", "lib", "build.js"));
    const publishCli = path.join(wpsjsRoot, "src", "index.js");
    const publishListPath = path.join(
        wpsjsRoot,
        "src",
        "lib",
        "publishlist.json",
    );
    const originalPublishList = fs.readFileSync(publishListPath);

    await buildModule.buildWithArgs({ pluginType: "offline" });
    await waitForArchive();

    try {
        fs.writeFileSync(publishListPath, "{}\n");
        run(process.execPath, [
            publishCli,
            "publish",
            "--serverUrl",
            serverUrl,
        ]);
    } finally {
        fs.writeFileSync(publishListPath, originalPublishList);
    }

    const publishList = readPublishList();
    if (publishList.length !== 1 || publishList[0].name !== project.name) {
        throw new Error(
            `Unexpected generated publish list: ${JSON.stringify(publishList)}`,
        );
    }
    if (
        publishList[0].online !== "false" ||
        publishList[0].url !== `${serverUrl}${project.name}.7z`
    ) {
        throw new Error(
            `Unexpected offline publish record: ${JSON.stringify(publishList[0])}`,
        );
    }

    fs.rmSync(deployRoot, { recursive: true, force: true });
    fs.mkdirSync(path.join(deployRoot, "addin"), { recursive: true });
    fs.copyFileSync(
        archivePath,
        path.join(deployRoot, "addin", path.basename(archivePath)),
    );
    fs.copyFileSync(publishHtmlPath, path.join(deployRoot, "publish.html"));

    process.stdout.write(
        `Verified publish record: ${JSON.stringify(publishList[0])}\n`,
    );
    process.stdout.write(
        `Staged offline installer: ${path.join(deployRoot, "publish.html")}\n`,
    );
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
