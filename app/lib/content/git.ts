import { execFileSync } from "node:child_process";

import { useContentCache } from "./cache";
import { repoRoot } from "./roots";
import type { GitAuthor, GitMetadata } from "./types";
const gitMetadataCache = new Map<string, GitMetadata | null>();

function normalizeDate(value: string | undefined): string | undefined {
  if (!value) {
    return undefined;
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) {
    return undefined;
  }

  return parsed.toISOString();
}

export function getGitMetadata(relativePath: string): GitMetadata | null {
  if (useContentCache) {
    const cached = gitMetadataCache.get(relativePath);
    if (cached !== undefined) {
      return cached;
    }
  }

  try {
    const output = execFileSync(
      "git",
      ["log", "--follow", "--format=%aI%x09%aN%x09%aE", "--", `docs/${relativePath}`],
      {
        cwd: repoRoot,
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"]
      }
    ).trim();

    if (!output) {
      if (useContentCache) {
        gitMetadataCache.set(relativePath, null);
      }
      return null;
    }

    const lines = output
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

    if (!lines.length) {
      if (useContentCache) {
        gitMetadataCache.set(relativePath, null);
      }
      return null;
    }

    const seenAuthors = new Set<string>();
    const authors: GitAuthor[] = [];

    for (const line of lines) {
      const [date, name, email] = line.split("\t");
      const authorKey = `${name}\u0000${email}`;
      if (name && !seenAuthors.has(authorKey)) {
        seenAuthors.add(authorKey);
        authors.push({
          name,
          email: email || undefined
        });
      }

      if (!date) {
        continue;
      }
    }

    const latest = lines[0]?.split("\t")[0];
    const earliest = lines.at(-1)?.split("\t")[0];
    const metadata: GitMetadata = {
      updatedAt: normalizeDate(latest),
      createdAt: normalizeDate(earliest),
      authors
    };

    if (useContentCache) {
      gitMetadataCache.set(relativePath, metadata);
    }
    return metadata;
  } catch {
    if (useContentCache) {
      gitMetadataCache.set(relativePath, null);
    }
    return null;
  }
}
