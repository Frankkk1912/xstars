declare module "wpsjs/vite_plugins" {
    import type { Plugin } from "vite";

    export function copyFile(options: { src: string; dest: string }): Plugin;
}
