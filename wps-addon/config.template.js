(function exposeInjectedConfig(root) {
  root.XSTARS_WPS_CONFIG = Object.freeze({
    port: Number("<port>"),
    token: "<token>",
    healthRetries: 3,
    retryIntervalMs: 150,
  });
})(typeof window === "undefined" ? globalThis : window);
