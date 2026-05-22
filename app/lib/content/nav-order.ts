import { useContentCache } from "./cache";
import { readMkdocsNavSources } from "./mkdocs-config";
import { baseMarkdownPath, sortNaturally } from "./source";

let mkdocsNavSourcesCache: string[] | null = null;

function getMkdocsNavSources(): string[] {
  if (!useContentCache || !mkdocsNavSourcesCache) {
    mkdocsNavSourcesCache = readMkdocsNavSources();
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
