import { defineConfig } from "vite"
import { copyFile } from "wpsjs/vite_plugins"

export default defineConfig({
    base: "./",
    plugins: [
        copyFile({ src: "manifest.xml", dest: "manifest.xml" }),
        copyFile({ src: "js", dest: "js" }),
        copyFile({ src: "images", dest: "images" }),
        copyFile({ src: "ribbon.xml", dest: "ribbon.xml" }),
    ],
    server: {
        host: "127.0.0.1",
    },
})
