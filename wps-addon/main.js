(function exposeAddin(root) {

  const CONTROL_COMMANDS = Object.freeze({
    xstarsRun: "run",
    xstarsQuickRun: "run_quick",
    xstarsWB: "run_wb",
    xstarsQPCR: "run_qpcr",
    xstarsCCK8: "run_cck8",
    xstarsELISA: "run_elisa",
    xstarsTransform: "run_transform_only",
    xstarsStandardCurve: "run_standard_curve",
    xstarsExport: "run_export",
    xstarsResetSettings: "run_reset_settings",
    xstarsBaseClassic: "run_set_base_theme_classic",
    xstarsBaseBW: "run_set_base_theme_bw",
    xstarsBaseMinimal: "run_set_base_theme_minimal",
    xstarsBaseDark: "run_set_base_theme_dark",
    xstarsThemeNone: "run_set_theme_none",
    xstarsThemeNature: "run_set_theme_nature",
    xstarsThemeScience: "run_set_theme_science",
    xstarsThemeCell: "run_set_theme_cell",
    xstarsThemeLancet: "run_set_theme_lancet",
    xstarsThemeNEJM: "run_set_theme_nejm",
    xstarsThemeJAMA: "run_set_theme_jama",
    xstarsThemeBMJ: "run_set_theme_bmj",
    xstarsJournalPaletteDefault: "run_set_journal_palette_default",
    xstarsJournalPaletteNature: "run_set_journal_palette_nature",
    xstarsJournalPaletteScience: "run_set_journal_palette_science",
    xstarsJournalPaletteCell: "run_set_journal_palette_cell",
    xstarsJournalPaletteLancet: "run_set_journal_palette_lancet",
    xstarsJournalPaletteNEJM: "run_set_journal_palette_nejm",
    xstarsJournalPaletteJAMA: "run_set_journal_palette_jama",
    xstarsJournalPaletteBMJ: "run_set_journal_palette_bmj",
    xstarsPaletteDefault: "run_set_palette_default",
    xstarsPaletteColorblind: "run_set_palette_colorblind",
    xstarsPaletteVibrant: "run_set_palette_vibrant",
    xstarsPalettePastel: "run_set_palette_pastel",
    xstarsPaletteDeep: "run_set_palette_deep",
    xstarsPaletteMuted: "run_set_palette_muted",
  });

  const SELECTION_COMMANDS = Object.freeze([
    "run",
    "run_quick",
    "run_wb",
    "run_qpcr",
    "run_cck8",
    "run_transform_only",
    "run_standard_curve",
  ]);
  const PICTURE_ID = /^XSTARS_[0-9]{8}_[0-9a-z]{8,32}$/;

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

  function reportError(application, modules, error) {
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

  async function runCommand(command, options) {
    const application = root.Application;
    let modules;
    try {
      modules = dependencies();
      modules.spreadsheet.setStatus(application, "XSTARS: 正在处理…");
      const supplied = options || {};
      const selection = "selection" in supplied
        ? supplied.selection
        : SELECTION_COMMANDS.includes(command)
          ? modules.spreadsheet.readSelection(application)
          : null;
      const response = await getClient(modules.service).command(
        command,
        selection,
        supplied.config || {},
        supplied.extra || {},
      );
      const writeback = modules.spreadsheet.executeWritebackPlan(
        application,
        response.writebackPlan,
      );
      return { response, selection, writeback };
    } catch (error) {
      return reportError(application, modules, error);
    }
  }

  async function runElisa() {
    const application = root.Application;
    let modules;
    try {
      modules = dependencies();
      modules.spreadsheet.setStatus(application, "XSTARS: 请选择 ELISA 标准品区域…");
      const standard = modules.spreadsheet.promptRange(
        application,
        "请框选 ELISA 标准品区域（首行为浓度表头）",
        "XSTARS ELISA — 标准品",
      );
      if (!standard) {
        modules.spreadsheet.setStatus(application, "XSTARS: ELISA 已取消");
        return { cancelled: true, stage: "standard" };
      }
      modules.spreadsheet.setStatus(application, "XSTARS: 请选择 ELISA 样本区域…");
      const sample = modules.spreadsheet.promptRange(
        application,
        "请框选 ELISA 样本区域（首行为分组表头）",
        "XSTARS ELISA — 样本",
      );
      if (!sample) {
        modules.spreadsheet.setStatus(application, "XSTARS: ELISA 已取消");
        return { cancelled: true, stage: "sample" };
      }
      return await runCommand("run_elisa", {
        selection: standard,
        extra: { sampleSelection: sample },
      });
    } catch (error) {
      return reportError(application, modules, error);
    }
  }

  async function runExport() {
    const application = root.Application;
    let modules;
    try {
      modules = dependencies();
      const shape = modules.spreadsheet.selectedShape(application);
      const format = modules.spreadsheet.promptText(
        application,
        "导出格式：png / tiff / jpg / pdf",
        "XSTARS 高分辨率导出",
        "png",
      );
      if (format === null) {
        modules.spreadsheet.setStatus(application, "XSTARS: 导出已取消");
        return { cancelled: true, stage: "format" };
      }
      const dpiText = modules.spreadsheet.promptText(
        application,
        "DPI（72-1200）",
        "XSTARS 高分辨率导出",
        "300",
      );
      if (dpiText === null) {
        modules.spreadsheet.setStatus(application, "XSTARS: 导出已取消");
        return { cancelled: true, stage: "dpi" };
      }
      const dpi = Number(dpiText);
      const shapeName = typeof shape.Name === "string" ? shape.Name : "";
      const exportRequest = { format: format.toLowerCase(), dpi };
      if (PICTURE_ID.test(shapeName)) {
        exportRequest.pictureId = shapeName;
        exportRequest.clipboard = false;
      } else {
        if (typeof shape.CopyPicture !== "function") {
          throw new Error("所选图片无法复制到剪贴板，且没有 XSTARS 重渲染数据");
        }
        shape.CopyPicture(2, -4147);
        exportRequest.clipboard = true;
      }
      const result = await runCommand("run_export", {
        selection: null,
        extra: { export: exportRequest },
      });
      if (result.response && result.response.export && typeof root.alert === "function") {
        root.alert(`XSTARS 导出完成\n${result.response.export.path}`);
      }
      return result;
    } catch (error) {
      return reportError(application, modules, error);
    }
  }

  function cancelActiveRequest() {
    return client ? client.cancelActiveRequest() : false;
  }

  function OnAddinLoad(ribbonUI) {
    if (root.Application) {
      root.Application.ribbonUI = ribbonUI;
    }
    return true;
  }

  function OnAction(control) {
    const command = control && CONTROL_COMMANDS[control.Id];
    if (command === "run_elisa") {
      void runElisa();
    } else if (command === "run_export") {
      void runExport();
    } else if (command) {
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
    runElisa,
    runExport,
  });
})(typeof window === "undefined" ? globalThis : window);
