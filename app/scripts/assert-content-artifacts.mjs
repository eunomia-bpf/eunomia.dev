import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const appDir = path.resolve(scriptDir, "..");
const generatedRoot = path.join(appDir, ".generated");

const requiredFiles = [
  path.join(generatedRoot, "content", "documents.json"),
  path.join(generatedRoot, "content", "content-model.json"),
  path.join(generatedRoot, "content", "manifest.json"),
  path.join(generatedRoot, "content", "site-sections.json"),
  path.join(generatedRoot, "search", "en.json"),
  path.join(generatedRoot, "search", "zh.json")
];

const missing = requiredFiles.filter((filePath) => !fs.existsSync(filePath));

if (missing.length) {
  console.error("Missing generated content artifacts:");
  for (const filePath of missing) {
    console.error(`- ${filePath}`);
  }
  console.error("Run `npm run generate:content-artifacts` or `npm run build` before starting the server.");
  process.exit(1);
}
