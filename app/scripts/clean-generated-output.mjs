import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const appDir = path.resolve(scriptDir, "..");

const generatedPaths = [
  ".static-builds",
  "out",
  path.join("public", "_content-assets"),
  path.join("public", "search-index"),
  path.join("public", "feed.xml"),
  path.join("public", "sitemap.xml"),
  path.join("public", "robots.txt"),
  path.join("public", "zh", "feed.xml"),
  path.join("public", "og", "default.svg")
];

function removeIfPresent(relativePath) {
  const absolutePath = path.join(appDir, relativePath);
  fs.rmSync(absolutePath, { recursive: true, force: true });
}

for (const relativePath of generatedPaths) {
  removeIfPresent(relativePath);
}

console.log(`Cleaned ${generatedPaths.length} generated output paths under ${appDir}`);
