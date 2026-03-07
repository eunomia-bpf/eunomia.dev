import path from "node:path";

import { visit } from "unist-util-visit";

import type { Locale } from "../site-data";
import { getDocsFileSet, getSiteFileSet } from "./fs-index";
import { resolveRouteFromDocSource } from "./manifest";
import {
  baseMarkdownPath,
  englishVariant,
  localizedVariant
} from "./source";

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

function splitSuffix(value: string): { pathname: string; search: string; hash: string } {
  const hashIndex = value.indexOf("#");
  const searchIndex = value.indexOf("?");

  const pathEnd =
    searchIndex >= 0 && hashIndex >= 0
      ? Math.min(searchIndex, hashIndex)
      : searchIndex >= 0
        ? searchIndex
        : hashIndex >= 0
          ? hashIndex
          : value.length;

  return {
    pathname: value.slice(0, pathEnd),
    search:
      searchIndex >= 0
        ? value.slice(searchIndex, hashIndex >= 0 && hashIndex > searchIndex ? hashIndex : undefined)
        : "",
    hash: hashIndex >= 0 ? value.slice(hashIndex) : ""
  };
}

function looksLikeAsset(relativePath: string): boolean {
  return assetExtensions.has(path.posix.extname(relativePath).toLowerCase());
}

function hasExplicitProtocol(value: string): boolean {
  return /^[a-z][a-z0-9+.-]*:/i.test(value);
}

function isSafeExternalUrl(value: string): boolean {
  const normalized = value.trim().toLowerCase();

  if (normalized.startsWith("http:") || normalized.startsWith("https:")) {
    return true;
  }

  if (normalized.startsWith("mailto:") || normalized.startsWith("tel:")) {
    return true;
  }

  return false;
}

export function toRawAssetPath(source: "docs" | "site", relativePath: string): string {
  return `/api/raw-assets/${source}/${relativePath}`;
}

function resolveDocLinkCandidate(relativePath: string): string | null {
  const docsFiles = getDocsFileSet();
  const candidates = new Set<string>();
  const normalized = path.posix.normalize(relativePath);

  if (normalized.endsWith(".md")) {
    candidates.add(baseMarkdownPath(normalized));
    candidates.add(englishVariant(normalized));
    candidates.add(localizedVariant(normalized, "zh"));
  } else {
    candidates.add(`${normalized}.md`);
    candidates.add(`${normalized}.en.md`);
    candidates.add(`${normalized}.zh.md`);
    candidates.add(path.posix.join(normalized, "README.md"));
    candidates.add(path.posix.join(normalized, "README.en.md"));
    candidates.add(path.posix.join(normalized, "README.zh.md"));
    candidates.add(path.posix.join(normalized, "index.md"));
    candidates.add(path.posix.join(normalized, "index.en.md"));
    candidates.add(path.posix.join(normalized, "index.zh.md"));
  }

  for (const candidate of candidates) {
    if (docsFiles.has(candidate)) {
      return candidate;
    }
  }

  return null;
}

function rewriteAbsolutePath(value: string): string {
  const { pathname, search, hash } = splitSuffix(value);
  const normalized = pathname.replace(/^\/+/, "");
  const docsFiles = getDocsFileSet();
  const siteFiles = getSiteFileSet();

  if (!normalized) {
    return `${pathname}${search}${hash}`;
  }

  if (docsFiles.has(normalized)) {
    const route = resolveRouteFromDocSource(normalized, normalized.endsWith(".zh.md") ? "zh" : "en");
    if (route) {
      return `${route}${search}${hash}`;
    }
  }

  if (docsFiles.has(normalized)) {
    return `${toRawAssetPath("docs", normalized)}${search}${hash}`;
  }

  if (siteFiles.has(normalized)) {
    return `${toRawAssetPath("site", normalized)}${search}${hash}`;
  }

  return `${pathname}${search}${hash}`;
}

function rewriteRelativePath(value: string, sourceRelativePath: string, locale: Locale): string {
  const { pathname, search, hash } = splitSuffix(value);
  const sourceDirectory = path.posix.dirname(baseMarkdownPath(sourceRelativePath));
  const resolved = path.posix.normalize(path.posix.join(sourceDirectory, pathname));
  const explicitLocale: Locale = pathname.endsWith(".zh.md") ? "zh" : locale;

  const docCandidate = resolveDocLinkCandidate(resolved);
  if (docCandidate) {
    const route = resolveRouteFromDocSource(docCandidate, explicitLocale);
    if (route) {
      return `${route}${search}${hash}`;
    }
  }

  if (getDocsFileSet().has(resolved)) {
    return `${toRawAssetPath("docs", resolved)}${search}${hash}`;
  }

  if (getSiteFileSet().has(resolved)) {
    return `${toRawAssetPath("site", resolved)}${search}${hash}`;
  }

  return `${pathname}${search}${hash}`;
}

function rewriteUrl(value: unknown, sourceRelativePath: string, locale: Locale): string | null {
  if (typeof value !== "string" || !value) {
    return null;
  }

  if (value.startsWith("#")) {
    return value;
  }

  if (hasExplicitProtocol(value)) {
    return isSafeExternalUrl(value) ? value : null;
  }

  if (value.startsWith("/")) {
    return rewriteAbsolutePath(value);
  }

  return rewriteRelativePath(value, sourceRelativePath, locale);
}

export function rewriteContentUrl(value: unknown, sourceRelativePath: string, locale: Locale): string | null {
  return rewriteUrl(value, sourceRelativePath, locale);
}

export function createRehypeRewriter(sourceRelativePath: string, locale: Locale) {
  return function rehypeRewriter() {
    return function rewriter(tree: unknown) {
      visit(tree as Parameters<typeof visit>[0], "element", (node: {
        properties?: {
          href?: unknown;
          src?: unknown;
          poster?: unknown;
        };
      }) => {
        if (!node.properties) {
          return;
        }

        if (typeof node.properties.href === "string") {
          const rewritten = rewriteUrl(node.properties.href, sourceRelativePath, locale);
          if (rewritten) {
            node.properties.href = rewritten;
          } else {
            delete node.properties.href;
          }
        }

        if (typeof node.properties.src === "string") {
          const rewritten = rewriteUrl(node.properties.src, sourceRelativePath, locale);
          if (rewritten) {
            node.properties.src = rewritten;
          } else {
            delete node.properties.src;
          }
        }

        if (typeof node.properties.poster === "string") {
          const rewritten = rewriteUrl(node.properties.poster, sourceRelativePath, locale);
          if (rewritten) {
            node.properties.poster = rewritten;
          } else {
            delete node.properties.poster;
          }
        }
      });
    };
  };
}
