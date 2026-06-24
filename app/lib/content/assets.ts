import fs from "node:fs";
import path from "node:path";

import { getDocsFileSet, getSiteFileSet } from "./fs-index";
import { appRoot, docsRoot, generatedContentDir, siteRoot } from "./roots";

export type StaticAssetSource = "docs" | "site";

export type StaticAssetEntry = {
  source: StaticAssetSource;
  relativePath: string;
  sourcePath: string;
  publicPath: string;
  outputPath: string;
};

type SerializedStaticAssetIndex = {
  generatedAt: string;
  assetBasePath: string;
  outputRoot: string;
  assets: Array<{
    source: StaticAssetSource;
    relativePath: string;
    publicPath: string;
  }>;
};

const assetExtensions = new Set([
  ".avif",
  ".gif",
  ".jpeg",
  ".jpg",
  ".json",
  ".pdf",
  ".png",
  ".svg",
  ".txt",
  ".webm",
  ".webp",
  ".xml",
  ".yml",
  ".yaml"
]);

const excludedStaticAssetPaths = new Set([
  // Large local AgentSight capture used during development. It is not linked
  // from the docs and should not be published as an indexable public asset.
  "docs:agentsight/sample-snapshot.json"
]);

export const staticAssetBasePath = "/_content-assets";
export const staticAssetOutputRoot = path.join(appRoot, "public", staticAssetBasePath.replace(/^\/+/, ""));
export const staticAssetIndexPath = path.join(generatedContentDir, "static-assets.json");

function normalizeRelativePath(relativePath: string): string {
  return path.posix.normalize(relativePath).replace(/^(\.\.(\/|\\|$))+/, "");
}

function rootFor(source: StaticAssetSource): string {
  return source === "docs" ? docsRoot : siteRoot;
}

export function isStaticAssetPath(relativePath: string): boolean {
  return assetExtensions.has(path.posix.extname(relativePath).toLowerCase());
}

function isExcludedStaticAsset(source: StaticAssetSource, relativePath: string): boolean {
  return excludedStaticAssetPaths.has(`${source}:${normalizeRelativePath(relativePath)}`);
}

export function getStaticAssetPublicPath(source: StaticAssetSource, relativePath: string): string {
  return `${staticAssetBasePath}/${source}/${normalizeRelativePath(relativePath)}`;
}

function buildStaticAssetEntries(
  source: StaticAssetSource,
  relativePaths: Iterable<string>,
  outputRoot: string
): StaticAssetEntry[] {
  const root = rootFor(source);

  return [...relativePaths]
    .filter((relativePath) => isStaticAssetPath(relativePath))
    .filter((relativePath) => !isExcludedStaticAsset(source, relativePath))
    .sort((left, right) =>
      left.localeCompare(right, "en", {
        numeric: true,
        sensitivity: "base"
      })
    )
    .map((relativePath) => {
      const normalized = normalizeRelativePath(relativePath);

      return {
        source,
        relativePath: normalized,
        sourcePath: path.join(root, normalized),
        publicPath: getStaticAssetPublicPath(source, normalized),
        outputPath: path.join(outputRoot, source, normalized)
      };
    });
}

export function getStaticAssetEntries(outputRoot: string = staticAssetOutputRoot): StaticAssetEntry[] {
  return [
    ...buildStaticAssetEntries("docs", getDocsFileSet(), outputRoot),
    ...buildStaticAssetEntries("site", getSiteFileSet(), outputRoot)
  ];
}

export function writeStaticAssets(
  outputRoot: string = staticAssetOutputRoot,
  indexPath: string = staticAssetIndexPath
) {
  const entries = getStaticAssetEntries(outputRoot);

  fs.rmSync(outputRoot, { recursive: true, force: true });
  fs.mkdirSync(outputRoot, { recursive: true });

  for (const entry of entries) {
    fs.mkdirSync(path.dirname(entry.outputPath), { recursive: true });
    fs.copyFileSync(entry.sourcePath, entry.outputPath);
  }

  fs.mkdirSync(path.dirname(indexPath), { recursive: true });
  const payload: SerializedStaticAssetIndex = {
    generatedAt: new Date().toISOString(),
    assetBasePath: staticAssetBasePath,
    outputRoot,
    assets: entries.map((entry) => ({
      source: entry.source,
      relativePath: entry.relativePath,
      publicPath: entry.publicPath
    }))
  };
  const tempPath = `${indexPath}.tmp`;
  fs.writeFileSync(tempPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  fs.renameSync(tempPath, indexPath);

  return {
    count: entries.length,
    outputRoot,
    indexPath
  };
}

export async function serveRawAsset(source: StaticAssetSource, pathSegments: string[]): Promise<{
  filePath: string;
  contentType: string;
} | null> {
  const relativePath = normalizeRelativePath(pathSegments.join("/"));
  const root = rootFor(source);
  const absolutePath = path.join(root, relativePath);
  const stats = fs.existsSync(absolutePath) ? fs.statSync(absolutePath) : null;

  if (!stats || stats.isDirectory()) {
    return null;
  }

  return {
    filePath: absolutePath,
    contentType: mimeTypeFor(absolutePath)
  };
}

function mimeTypeFor(filePath: string): string {
  const extension = path.extname(filePath).toLowerCase();
  switch (extension) {
    case ".css":
      return "text/css; charset=utf-8";
    case ".gif":
      return "image/gif";
    case ".html":
      return "text/html; charset=utf-8";
    case ".jpeg":
    case ".jpg":
      return "image/jpeg";
    case ".json":
      return "application/json; charset=utf-8";
    case ".pdf":
      return "application/pdf";
    case ".png":
      return "image/png";
    case ".svg":
      return "image/svg+xml";
    case ".txt":
    case ".md":
      return "text/plain; charset=utf-8";
    case ".webm":
      return "video/webm";
    case ".webp":
      return "image/webp";
    case ".xml":
      return "application/xml; charset=utf-8";
    case ".yml":
    case ".yaml":
      return "text/yaml; charset=utf-8";
    default:
      return "application/octet-stream";
  }
}
