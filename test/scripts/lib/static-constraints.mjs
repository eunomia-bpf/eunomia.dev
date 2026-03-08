import fs from "node:fs";
import path from "node:path";

const SOURCE_EXTENSIONS = new Set([".cjs", ".cts", ".js", ".jsx", ".mjs", ".mts", ".ts", ".tsx"]);
const IGNORED_DIRS = new Set([".generated", ".static-builds", "node_modules", "out", "public"]);

function shouldIgnoreDirectory(directoryName) {
  return IGNORED_DIRS.has(directoryName) || directoryName.startsWith(".next");
}

function walkFiles(rootDir, results = []) {
  for (const entry of fs.readdirSync(rootDir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      if (shouldIgnoreDirectory(entry.name)) {
        continue;
      }
      walkFiles(path.join(rootDir, entry.name), results);
      continue;
    }

    const extension = path.extname(entry.name).toLowerCase();
    if (SOURCE_EXTENSIONS.has(extension)) {
      results.push(path.join(rootDir, entry.name));
    }
  }

  return results;
}

function relativeTo(rootDir, filePath) {
  return path.relative(rootDir, filePath).split(path.sep).join("/");
}

export function getApiRouteFiles(appDir) {
  const apiDir = path.join(appDir, "pages", "api");
  if (!fs.existsSync(apiDir)) {
    return [];
  }

  return walkFiles(apiDir).map((filePath) => relativeTo(appDir, filePath)).sort();
}

export function getServerSidePropsFiles(appDir) {
  return walkFiles(appDir)
    .filter((filePath) => fs.readFileSync(filePath, "utf8").includes("getServerSideProps"))
    .map((filePath) => relativeTo(appDir, filePath))
    .sort();
}

export function getStaticConstraintViolations(appDir) {
  return {
    apiRouteFiles: getApiRouteFiles(appDir),
    getServerSidePropsFiles: getServerSidePropsFiles(appDir)
  };
}

export function formatStaticConstraintFailures(violations) {
  const failures = [];

  if (violations.apiRouteFiles.length) {
    failures.push(`pages/api route files remain: ${violations.apiRouteFiles.join(", ")}`);
  }

  if (violations.getServerSidePropsFiles.length) {
    failures.push(`getServerSideProps remains in: ${violations.getServerSidePropsFiles.join(", ")}`);
  }

  return failures;
}

export function assertStaticConstraintViolations(appDir) {
  const violations = getStaticConstraintViolations(appDir);
  const failures = formatStaticConstraintFailures(violations);

  if (failures.length) {
    throw new Error(failures.join("\n"));
  }

  return violations;
}
