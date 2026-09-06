import { defineConfig } from "vite";
import { copyFile } from "wpsjs/vite_plugins";

export default defineConfig({
  base: "./",
  plugins: [
    copyFile({ src: "manifest.xml", dest: "manifest.xml" }),
    copyFile({ src: "ribbon.xml", dest: "ribbon.xml" }),
    copyFile({ src: "config.js", dest: "config.js" }),
    copyFile({ src: "service-client.js", dest: "service-client.js" }),
    copyFile({ src: "spreadsheet.js", dest: "spreadsheet.js" }),
    copyFile({ src: "main.js", dest: "main.js" }),
    copyFile({ src: "assets", dest: "assets" }),
  ],
  server: {
    host: "127.0.0.1",
    port: 3889,
  },
});
