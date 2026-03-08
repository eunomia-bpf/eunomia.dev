import fs from "node:fs";
import path from "node:path";

const appDir = process.cwd();
const distDir = path.resolve(appDir, process.env.NEXT_DIST_DIR ?? ".next");
const outDir = path.resolve(appDir, "out");
const publicDir = path.resolve(appDir, "public");

const distIndexPath = path.join(distDir, "index.html");

function syncPublicDir() {
  if (!fs.existsSync(publicDir)) {
    return;
  }

  fs.cpSync(publicDir, outDir, { recursive: true });
}

if (!fs.existsSync(distIndexPath)) {
  console.error(`No static export detected in ${outDir} or ${distDir}`);
  process.exit(1);
}

if (outDir !== distDir) {
  fs.rmSync(outDir, { recursive: true, force: true });
  fs.cpSync(distDir, outDir, { recursive: true });
}

syncPublicDir();

console.log(`Copied static export from ${distDir} to ${outDir}`);
