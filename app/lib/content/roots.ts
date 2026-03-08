import path from "node:path";
import { fileURLToPath } from "node:url";

function normalizeAbsolutePath(value: string): string {
  return path.resolve(value);
}

function resolveConfiguredPath(envName: string, fallbackPath: string): string {
  const configured = process.env[envName];
  return normalizeAbsolutePath(configured || fallbackPath);
}

const currentDir = path.dirname(fileURLToPath(import.meta.url));

export const appRoot = resolveConfiguredPath("EUNOMIA_APP_ROOT", path.resolve(currentDir, "..", ".."));
export const repoRoot = resolveConfiguredPath("EUNOMIA_REPO_ROOT", path.resolve(appRoot, ".."));
export const docsRoot = resolveConfiguredPath("EUNOMIA_DOCS_ROOT", path.join(repoRoot, "docs"));
export const siteRoot = resolveConfiguredPath("EUNOMIA_SITE_ROOT", path.join(repoRoot, "site"));
export const generatedRoot = resolveConfiguredPath("EUNOMIA_GENERATED_ROOT", path.join(appRoot, ".generated"));
export const generatedContentDir = path.join(generatedRoot, "content");
export const generatedSearchDir = path.join(generatedRoot, "search");
export const mkdocsConfigPath = resolveConfiguredPath(
  "EUNOMIA_MKDOCS_CONFIG",
  path.join(repoRoot, "mkdocs.yaml")
);
