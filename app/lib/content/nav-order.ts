import fs from "node:fs";

import { useContentCache } from "./cache";
import { mkdocsConfigPath } from "./roots";
import { baseMarkdownPath, sortNaturally } from "./source";
const navSourcePattern = /^\s*-\s+(?:[^:]+:\s+)?(.+?\.md)\s*$/;

let mkdocsNavSourcesCache: string[] | null = null;

function extractMkdocsNavSources(): string[] {
  const content = fs.readFileSync(mkdocsConfigPath, "utf8");
  const orderedSources: string[] = [];

  for (const line of content.split(/\r?\n/)) {
    if (line.includes("... |")) {
      continue;
    }

    const match = line.match(navSourcePattern);
    if (!match) {
      continue;
    }

    const candidate = match[1]?.trim();
    if (!candidate) {
      continue;
    }

    orderedSources.push(baseMarkdownPath(candidate));
  }

  return [...new Set(orderedSources)];
}

function getMkdocsNavSources(): string[] {
  if (!useContentCache || !mkdocsNavSourcesCache) {
    mkdocsNavSourcesCache = extractMkdocsNavSources();
  }

  return mkdocsNavSourcesCache;
}

export function orderSourcesByMkdocsNav(sources: string[]): string[] {
  const normalizedSources = [...new Set(sources.map((source) => baseMarkdownPath(source)))];
  const available = new Set(normalizedSources);
  const navOrdered = getMkdocsNavSources().filter((source) => available.has(source));
  const navSet = new Set(navOrdered);
  const remainder = sortNaturally(normalizedSources.filter((source) => !navSet.has(source)));
  return [...navOrdered, ...remainder];
}
