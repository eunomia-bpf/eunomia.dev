import fs from "node:fs";
import path from "node:path";

import { docsRoot, siteRoot } from "./fs-index";

export async function serveRawAsset(source: "docs" | "site", pathSegments: string[]): Promise<{
  filePath: string;
  contentType: string;
} | null> {
  const root = source === "docs" ? docsRoot : siteRoot;
  const relativePath = path.posix.normalize(pathSegments.join("/")).replace(/^(\.\.(\/|\\|$))+/, "");
  const absolutePath = path.join(root, relativePath);

  if (!absolutePath.startsWith(root) || !fs.existsSync(absolutePath) || fs.statSync(absolutePath).isDirectory()) {
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
