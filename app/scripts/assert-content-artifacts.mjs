import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const appDir = path.resolve(scriptDir, "..");
const generatedRoot = path.join(appDir, ".generated");
const staticBuildRoot = path.join(appDir, "out");

const requiredFiles = [
  path.join(generatedRoot, "content", "documents.json"),
  path.join(generatedRoot, "content", "content-model.json"),
  path.join(generatedRoot, "content", "manifest.json"),
  path.join(generatedRoot, "content", "static-assets.json"),
  path.join(generatedRoot, "content", "static-metadata.json"),
  path.join(generatedRoot, "content", "site-sections.json"),
  path.join(generatedRoot, "search", "en.json"),
  path.join(generatedRoot, "search", "zh.json"),
  path.join(appDir, "public", "feed.xml"),
  path.join(appDir, "public", "zh", "feed.xml"),
  path.join(appDir, "public", "sitemap.xml"),
  path.join(appDir, "public", "robots.txt"),
  path.join(appDir, "public", "og", "default.svg"),
  path.join(staticBuildRoot, "index.html"),
  path.join(staticBuildRoot, "feed.xml"),
  path.join(staticBuildRoot, "robots.txt"),
  path.join(staticBuildRoot, "sitemap.xml"),
  path.join(staticBuildRoot, "zh", "feed.xml")
];

const requiredDirs = [
  path.join(appDir, "public", "_content-assets"),
  path.join(staticBuildRoot, "_content-assets"),
  path.join(staticBuildRoot, "search-index"),
  path.join(staticBuildRoot, "zh")
];

const missing = requiredFiles.filter((filePath) => !fs.existsSync(filePath));
const missingDirs = requiredDirs.filter((directoryPath) => !fs.existsSync(directoryPath));

if (missing.length || missingDirs.length) {
  console.error("Missing generated content artifacts:");
  for (const filePath of missing) {
    console.error(`- ${filePath}`);
  }
  for (const directoryPath of missingDirs) {
    console.error(`- ${directoryPath}`);
  }
  console.error("Run `npm run build` before starting the static preview server.");
  process.exit(1);
}
