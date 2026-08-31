(function exposeServiceClient(root) {

  const SCHEMA_VERSION = "1.0";
  const DEFAULT_RETRIES = 3;
  const DEFAULT_RETRY_INTERVAL_MS = 150;

  class ServiceError extends Error {
    constructor(code, message, status) {
      super(message);
      this.name = "ServiceError";
      this.code = code;
      this.status = status || 0;
    }
  }

  const USER_MESSAGES = Object.freeze({
    CONFIG_MISSING: "XSTARS WPS 配置未生成，请重新运行安装器或配置注入脚本。",
    SERVICE_UNAVAILABLE: "无法连接 XSTARS 本地服务，请确认服务已启动。",
    UNAUTHORIZED: "XSTARS 本地服务鉴权失败，请重新安装或刷新加载项配置。",
    ORIGIN_DENIED: "XSTARS 本地服务拒绝了当前加载项来源。",
    INVALID_REQUEST: "XSTARS 请求格式无效。",
    INVALID_COMMAND: "当前 XSTARS 命令不可用。",
    INVALID_SELECTION: "选区无效，请选择含表头的单个连续数据区域。",
    PAYLOAD_TOO_LARGE: "选区过大，请将数据限制在 200 行、200 列以内。",
    INVALID_PATH: "XSTARS 返回了不允许的文件路径。",
    CANCELLED: "操作已取消。",
    BUSY: "XSTARS 正在处理另一个任务，请稍后重试。",
    TIMEOUT: "XSTARS 分析超时，请缩小选区后重试。",
    ANALYSIS_FAILED: "XSTARS 无法分析当前数据，请检查数据格式。",
    PAYLOAD_MISSING: "找不到该图的高清重渲染数据，请重新生成图表；普通图片可重试剪贴板导出。",
    PAYLOAD_CORRUPT: "该图的高清重渲染数据已损坏，请重新生成图表。",
    PAYLOAD_VERSION: "该图由不兼容版本生成，请使用当前版本重新生成图表。",
    EXPORT_FORMAT: "导出格式无效，仅支持 PNG、TIFF、JPG、PDF。",
    EXPORT_DPI: "导出 DPI 无效，请输入 72 到 1200 的整数。",
    EXPORT_PATH: "无法写入 XSTARS 受控导出目录。",
    INTERNAL_ERROR: "XSTARS 本地服务发生内部错误。",
    INVALID_RESPONSE: "XSTARS 本地服务返回了无效响应。",
  });

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function validateConfig(config) {
    if (!config || typeof config !== "object") {
      throw new ServiceError("CONFIG_MISSING", "WPS service config is missing");
    }
    const token = config.token;
    if (
      typeof token !== "string" ||
      token.length < 32 ||
      token.includes("<token>")
    ) {
      throw new ServiceError("CONFIG_MISSING", "WPS service token is not injected");
    }
    const rawPorts = Array.isArray(config.ports) ? config.ports : [config.port];
    const ports = rawPorts.map(Number);
    if (
      ports.length === 0 ||
      ports.some((port) => !Number.isInteger(port) || port < 1 || port > 65535)
    ) {
      throw new ServiceError("CONFIG_MISSING", "WPS service port is not injected");
    }
    return {
      token,
      ports: [...new Set(ports)],
      healthRetries: Number.isInteger(config.healthRetries)
        ? Math.max(1, config.healthRetries)
        : DEFAULT_RETRIES,
      retryIntervalMs: Number.isFinite(config.retryIntervalMs)
        ? Math.max(0, config.retryIntervalMs)
        : DEFAULT_RETRY_INTERVAL_MS,
    };
  }

  async function readJsonResponse(fetchImpl, url, init) {
    let response;
    try {
      response = await fetchImpl(url, init);
    } catch (error) {
      if (error && error.name === "AbortError") {
        throw new ServiceError("CANCELLED", "request aborted");
      }
      throw new ServiceError("SERVICE_UNAVAILABLE", String(error && error.message));
    }
    const text = await response.text();
    let payload;
    try {
      payload = JSON.parse(text);
    } catch {
      throw new ServiceError("INVALID_RESPONSE", "response is not valid JSON", response.status);
    }
    return { response, payload };
  }

  function responseError(response, payload) {
    const remote = payload && payload.error;
    const code = remote && typeof remote.code === "string"
      ? remote.code
      : response.status === 401
        ? "UNAUTHORIZED"
        : `HTTP_${response.status}`;
    const message = remote && typeof remote.message === "string"
      ? remote.message
      : `HTTP ${response.status}`;
    return new ServiceError(code, message, response.status);
  }

  class WpsServiceClient {
    constructor(config, options) {
      this.config = validateConfig(config);
      this.fetchImpl = (options && options.fetch) || root.fetch;
      this.sleepImpl = (options && options.sleep) || sleep;
      if (typeof this.fetchImpl !== "function") {
        throw new ServiceError("SERVICE_UNAVAILABLE", "fetch is unavailable");
      }
      this.baseUrl = null;
      this.activeController = null;
    }

    async discoverService() {
      for (let attempt = 0; attempt < this.config.healthRetries; attempt += 1) {
        for (const port of this.config.ports) {
          const baseUrl = `http://127.0.0.1:${port}`;
          try {
            const { response, payload } = await readJsonResponse(
              this.fetchImpl,
              `${baseUrl}/health`,
              { method: "GET" },
            );
            if (
              response.ok &&
              payload &&
              payload.ok === true &&
              payload.service === "xstars-wps-service" &&
              Number(payload.port) === port
            ) {
              this.baseUrl = baseUrl;
              return payload;
            }
          } catch (error) {
            if (error.code === "CANCELLED") {
              throw error;
            }
          }
        }
        if (attempt + 1 < this.config.healthRetries) {
          await this.sleepImpl(this.config.retryIntervalMs);
        }
      }
      throw new ServiceError("SERVICE_UNAVAILABLE", "health checks failed");
    }

    cancelActiveRequest() {
      if (!this.activeController) {
        return false;
      }
      this.activeController.abort();
      return true;
    }

    async command(command, selection, config, extra, options) {
      if (!this.baseUrl) {
        await this.discoverService();
      }
      const controller = new AbortController();
      const externalSignal = options && options.signal;
      const abortFromExternal = () => controller.abort();
      if (externalSignal) {
        if (externalSignal.aborted) {
          controller.abort();
        } else {
          externalSignal.addEventListener("abort", abortFromExternal, { once: true });
        }
      }
      this.activeController = controller;
      try {
        const body = {
          version: SCHEMA_VERSION,
          command,
          config: config || {},
        };
        if (selection) {
          body.selection = selection;
        }
        const additional = extra || {};
        if (additional.sampleSelection) {
          body.sampleSelection = additional.sampleSelection;
        }
        if (additional.export) {
          body.export = additional.export;
        }
        const { response, payload } = await readJsonResponse(
          this.fetchImpl,
          `${this.baseUrl}/command`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${this.config.token}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify(body),
            signal: controller.signal,
          },
        );
        if (!response.ok || !payload || payload.ok !== true) {
          throw responseError(response, payload);
        }
        if (!payload.writebackPlan || typeof payload.writebackPlan !== "object") {
          throw new ServiceError("INVALID_RESPONSE", "writebackPlan is missing", response.status);
        }
        return payload;
      } finally {
        if (externalSignal) {
          externalSignal.removeEventListener("abort", abortFromExternal);
        }
        if (this.activeController === controller) {
          this.activeController = null;
        }
      }
    }
  }

  function toUserMessage(error) {
    const stableMessage = error && USER_MESSAGES[error.code]
      ? USER_MESSAGES[error.code]
      : USER_MESSAGES.SERVICE_UNAVAILABLE;
    const rawDetail = error && typeof error.detail === "string"
      ? error.detail
      : error && typeof error.message === "string"
        ? error.message
        : "";
    const detail = rawDetail.replace(/\s+/g, " ").trim().slice(0, 200);
    if (detail && detail !== stableMessage) {
      return `${stableMessage}\n详情：${detail}`;
    }
    return stableMessage;
  }

  root.XstarsServiceClient = Object.freeze({
    SCHEMA_VERSION,
    ServiceError,
    WpsServiceClient,
    toUserMessage,
    validateConfig,
  });
})(typeof window === "undefined" ? globalThis : window);
