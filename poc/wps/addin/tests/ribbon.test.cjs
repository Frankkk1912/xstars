const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const ribbonSource = fs.readFileSync(
        path.join(__dirname, "..", "js", "ribbon.js"),
        "utf8",
);

function loadRibbon(workbookName = null) {
        const alerts = [];
        const context = {
                window: {
                        alert: (message) => alerts.push(message),
                        location: { origin: "http://127.0.0.1:3890" },
                        Application: {
                                ActiveWorkbook: workbookName
                                        ? { Name: workbookName }
                                        : null,
                        },
                },
        };

        vm.createContext(context);
        vm.runInContext(ribbonSource, context);
        return { alerts, context };
}

test("OnAddinLoad stores the WPS Ribbon object", () => {
        const { context } = loadRibbon();
        const ribbonUI = { Invalidate: () => {} };

        assert.equal(context.OnAddinLoad(ribbonUI), true);
        assert.equal(context.window.Application.ribbonUI, ribbonUI);
});

test("Gate 0 button proves the callback and reports the workbook", () => {
        const { alerts, context } = loadRibbon("gate0.xlsx");

        assert.equal(context.OnAction({ Id: "xstarsGate0Callback" }), true);
        assert.deepEqual(alerts, [
                "XSTARS Gate 0 回调成功\n工作簿：gate0.xlsx\nOrigin：http://127.0.0.1:3890",
        ]);
});

test("Unrelated controls are ignored", () => {
        const { alerts, context } = loadRibbon("gate0.xlsx");

        assert.equal(context.OnAction({ Id: "otherControl" }), true);
        assert.deepEqual(alerts, []);
});
