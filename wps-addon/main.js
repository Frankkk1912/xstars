(function exposeAddin(root) {

  const CONTROL_COMMANDS = Object.freeze({
    xstarsRun: "run",
    xstarsQuickRun: "run_quick",
  });

  let client = null;

  function dependencies() {
    const service = root.XstarsServiceClient;
    const spreadsheet = root.XstarsSpreadsheet;
    if (!service || !spreadsheet) {
      throw new Error("XSTARS 加载项脚本不完整，请重新安装加载项。");
    }
    return { service, spreadsheet };
  }

  function getClient(service) {
    if (!client) {
      client = new service.WpsServiceClient(root.XSTARS_WPS_CONFIG);
    }
    return client;
  }

  async function runCommand(command) {
    const application = root.Application;
    let modules;
    try {
      modules = dependencies();
      modules.spreadsheet.setStatus(application, "XSTARS: 正在分析…");
      const selection = modules.spreadsheet.readSelection(application);
      const response = await getClient(modules.service).command(
        command,
        selection,
        {},
      );
      const writeback = modules.spreadsheet.executeWritebackPlan(
        application,
        response.writebackPlan,
      );
      return { response, selection, writeback };
    } catch (error) {
      const message = modules && error && error.code
        ? modules.service.toUserMessage(error)
        : String(error && error.message ? error.message : error);
      if (modules) {
        modules.spreadsheet.showError(application, message);
      } else if (typeof root.alert === "function") {
        root.alert(message);
      }
      return { error, message };
    }
  }

  function cancelActiveRequest() {
    if (!client) {
      return false;
    }
    return client.cancelActiveRequest();
  }

  function OnAddinLoad(ribbonUI) {
    if (root.Application) {
      root.Application.ribbonUI = ribbonUI;
    }
    return true;
  }

  function OnAction(control) {
    const command = control && CONTROL_COMMANDS[control.Id];
    if (command) {
      void runCommand(command);
    }
    return true;
  }

  function GetImage(control) {
    return control && control.Id === "xstarsQuickRun"
      ? "assets/quick-run.svg"
      : "assets/run.svg";
  }

  root.OnAddinLoad = OnAddinLoad;
  root.OnAction = OnAction;
  root.GetImage = GetImage;
  root.XstarsWpsAddin = Object.freeze({
    CONTROL_COMMANDS,
    cancelActiveRequest,
    runCommand,
  });
})(typeof window === "undefined" ? globalThis : window);
