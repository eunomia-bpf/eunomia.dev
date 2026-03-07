import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(process.cwd(), "..");
export const docsRoot = path.join(repoRoot, "docs");
export const siteRoot = path.join(repoRoot, "site");

const excludedGenericSections = new Set(["assets", "img", "blog", "blogs", "tutorials"]);

let docsFileSetCache: Set<string> | null = null;
let siteFileSetCache: Set<string> | null = null;
let topLevelSectionsCache: string[] | null = null;

function toPosix(value: string): string {
  return value.split(path.sep).join("/");
}

function walkFiles(root: string): string[] {
  const queue = [root];
  const files: string[] = [];

  while (queue.length) {
    const current = queue.pop();
    if (!current) {
      continue;
    }

    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        queue.push(fullPath);
        continue;
      }
      files.push(fullPath);
    }
  }

  return files;
}

export function getDocsFileSet(): Set<string> {
  if (!docsFileSetCache) {
    docsFileSetCache = new Set(
      walkFiles(docsRoot).map((filePath) => toPosix(path.relative(docsRoot, filePath)))
    );
  }
  return docsFileSetCache;
}

export function getSiteFileSet(): Set<string> {
  if (!siteFileSetCache) {
    siteFileSetCache = new Set(
      walkFiles(siteRoot).map((filePath) => toPosix(path.relative(siteRoot, filePath)))
    );
  }
  return siteFileSetCache;
}

export function getTopLevelSections(): string[] {
  if (!topLevelSectionsCache) {
    topLevelSectionsCache = fs
      .readdirSync(docsRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .filter((name) => !excludedGenericSections.has(name))
      .sort((left, right) =>
        left.localeCompare(right, "en", {
          numeric: true,
          sensitivity: "base"
        })
      );
  }

  return topLevelSectionsCache;
}
