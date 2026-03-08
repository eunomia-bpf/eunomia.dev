import fs from "node:fs";
import path from "node:path";

import { docsRoot, siteRoot } from "./roots";

export async function serveRawAsset(source: "docs" | "site", pathSegments: string[]): Promise<{
  filePath: string;
  contentType: string;
} | null> {
  const root = source === "docs" ? docsRoot : siteRoot;
  const realRoot = fs.realpathSync(root);
  const relativePath = path.posix.normalize(pathSegments.join("/")).replace(/^(\.\.(\/|\\|$))+/, "");
  const absolutePath = path.join(root, relativePath);
  const stats = fs.existsSync(absolutePath) ? fs.statSync(absolutePath) : null;

  if (!absolutePath.startsWith(root) || !stats || stats.isDirectory()) {
    return null;
  }

  const realPath = fs.realpathSync(absolutePath);
  const relativeToRoot = path.relative(realRoot, realPath);
  if (relativeToRoot.startsWith("..") || path.isAbsolute(relativeToRoot)) {
    return null;
  }

  return {
    filePath: realPath,
    contentType: mimeTypeFor(realPath)
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
