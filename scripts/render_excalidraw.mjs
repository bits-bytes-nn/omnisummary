#!/usr/bin/env node
// Render an .excalidraw file to PNG with the hand-drawn (Virgil/Excalifont) font intact.
//
// The `excalidraw-to-png` CLI rasterizes via resvg, which ignores the font that
// Excalidraw embeds in the SVG as an @font-face data: URL and falls back to a plain
// sans-serif. We instead let Excalidraw export the SVG (fonts embedded), then have a
// headless browser screenshot it — the browser honors the embedded woff2, so the
// output matches what the Excalidraw app produces.
//
// Usage: node scripts/render_excalidraw.mjs <input.excalidraw> <output.png> [scale]
import { readFileSync } from "fs";
import { resolve } from "path";
import { execFileSync } from "child_process";
import { createRequire } from "module";

const TOOL_DIR = "/Users/youngmki/Projects/tools/excalidraw-to-png";
const TOOL = `${TOOL_DIR}/cli.mjs`;
const require = createRequire(`${TOOL_DIR}/`);
const { chromium } = require("playwright");

async function main() {
  const [input, output, scaleArg] = process.argv.slice(2);
  if (!input || !output) {
    console.error("Usage: render_excalidraw.mjs <input.excalidraw> <output.png> [scale]");
    process.exit(1);
  }
  const scale = Number(scaleArg || 2);

  const svgPath = "/tmp/_excalidraw_render.svg";
  execFileSync("node", [TOOL, resolve(input), svgPath, "--svg"], { stdio: "inherit" });
  const svg = readFileSync(svgPath, "utf-8");

  const m = svg.match(/width="([0-9.]+)"[\s\S]*?height="([0-9.]+)"/);
  const w = Math.ceil(parseFloat(m[1]));
  const h = Math.ceil(parseFloat(m[2]));

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    viewport: { width: w * scale, height: h * scale },
    deviceScaleFactor: scale,
  });
  const html = `<!DOCTYPE html><html><head><meta charset=utf-8>
<style>html,body{margin:0;padding:0;background:#fff}svg{display:block}</style>
</head><body>${svg}</body></html>`;
  await page.setContent(html, { waitUntil: "networkidle" });
  await page.waitForTimeout(500);
  const el = await page.$("svg");
  await el.screenshot({ path: resolve(output) });
  await browser.close();
  console.error(`PNG saved to ${output}`);
}

main().catch((e) => {
  console.error(`Error: ${e.message}`);
  process.exit(1);
});
