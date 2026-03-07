import fs from "node:fs";
import path from "node:path";

import matter from "gray-matter";

import { useContentCache } from "./cache";
import { docsRoot } from "./fs-index";
import { slugifyTitle } from "./source";
import type { ParsedMarkdown } from "./types";

const markdownCache = new Map<string, ParsedMarkdown>();

export type UnsupportedMarkdownConstruct = {
  kind: string;
  line: number;
  snippet: string;
};

function readFile(relativePath: string, root: string = docsRoot): string {
  return fs.readFileSync(path.join(root, relativePath), "utf8");
}

function removeLeadingHeading(markdown: string): { titleFromHeading?: string; body: string } {
  const lines = markdown.replace(/^\uFEFF/, "").split("\n");
  let index = 0;

  while (index < lines.length && !lines[index]?.trim()) {
    index += 1;
  }

  const firstContentLine = lines[index];
  const headingMatch = firstContentLine?.match(/^#\s+(.+?)\s*$/);
  if (!headingMatch) {
    return { body: markdown.trim() };
  }

  const titleFromHeading = stripInlineMarkdown(headingMatch[1]);
  lines.splice(index, 1);
  while (index < lines.length && !lines[index]?.trim()) {
    lines.splice(index, 1);
  }

  const body = lines.join("\n").trim();
  return { titleFromHeading, body };
}

function stripInlineMarkdown(value: string): string {
  return value
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[*_~>#]/g, "")
    .replace(/:[a-z0-9-]+:/gi, "")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeMarkdown(markdown: string): string {
  return markdown
    .replace(/\r\n?/g, "\n")
    .replace(/<!--\s*more\s*-->/gi, "\n")
    .replace(/(!?\[[^\]]*]\([^)]+\))\{[^}\n]+\}/g, "$1")
    .replace(/(#+[^\n]+?)\s*\{#[^}\n]+\}/g, "$1")
    .replace(/:[a-z0-9-]+:/gi, "")
    .trim();
}

function collapseWhitespace(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

const unsupportedConstructMatchers: Array<{
  kind: string;
  pattern: RegExp;
}> = [
  {
    kind: "mkdocs-snippet-include",
    pattern: /^--8<--/
  },
  {
    kind: "generic-directive-block",
    pattern: /^:::\s+\S+/
  }
];

export function findUnsupportedMarkdownConstructs(markdown: string): UnsupportedMarkdownConstruct[] {
  const constructs: UnsupportedMarkdownConstruct[] = [];
  let inFence = false;

  for (const [index, line] of markdown.replace(/\r\n?/g, "\n").split("\n").entries()) {
    const trimmed = line.trim();
    if (/^(```|~~~)/.test(trimmed)) {
      inFence = !inFence;
      continue;
    }

    if (inFence || !trimmed) {
      continue;
    }

    for (const matcher of unsupportedConstructMatchers) {
      if (!matcher.pattern.test(trimmed)) {
        continue;
      }

      constructs.push({
        kind: matcher.kind,
        line: index + 1,
        snippet: trimmed
      });
    }
  }

  return constructs;
}

export function assertSupportedMarkdown(markdown: string, relativePath: string) {
  const unsupported = findUnsupportedMarkdownConstructs(markdown);
  if (!unsupported.length) {
    return;
  }

  const summary = unsupported
    .map((construct) => `${construct.kind} at line ${construct.line}: ${construct.snippet}`)
    .join("; ");

  throw new Error(`Unsupported Markdown construct in ${relativePath}: ${summary}`);
}

export type MarkdownHeading = {
  id: string;
  text: string;
  depth: number;
};

export function extractMarkdownHeadings(markdown: string): MarkdownHeading[] {
  const headings: MarkdownHeading[] = [];
  let inFence = false;

  for (const line of markdown.split("\n")) {
    const trimmed = line.trim();
    if (/^(```|~~~)/.test(trimmed)) {
      inFence = !inFence;
      continue;
    }

    if (inFence) {
      continue;
    }

    const match = trimmed.match(/^(#{2,6})\s+(.+?)\s*$/);
    if (!match) {
      continue;
    }

    const explicitId = match[2].match(/\{#([^}\n]+)\}\s*$/)?.[1];
    const text = stripInlineMarkdown(match[2].replace(/\s*\{#[^}\n]+\}\s*$/, ""));
    if (!text) {
      continue;
    }

    headings.push({
      id: explicitId ?? slugifyTitle(text),
      text,
      depth: match[1].length
    });
  }

  return headings;
}

export function markdownToSearchText(markdown: string): string {
  return collapseWhitespace(
    normalizeMarkdown(markdown)
      .replace(/```[\s\S]*?```/g, (block) => block.replace(/```[^\n]*\n?/g, " "))
      .replace(/`([^`]+)`/g, "$1")
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1")
      .replace(/<\/?[^>]+>/g, " ")
      .replace(/[*_~>#-]/g, " ")
  );
}

function makeExcerpt(markdown: string): string {
  const blocks = markdown
    .split(/\n{2,}/)
    .map((block) => collapseWhitespace(stripInlineMarkdown(block)))
    .filter(Boolean);

  const preferredBlock = blocks.find((block) => block.length > 40) ?? blocks[0] ?? "";
  return preferredBlock.slice(0, 220);
}

function parseDate(rawValue: unknown): string | undefined {
  if (!rawValue) {
    return undefined;
  }

  if (rawValue instanceof Date && !Number.isNaN(rawValue.valueOf())) {
    return rawValue.toISOString().slice(0, 10);
  }

  if (typeof rawValue === "string") {
    const parsed = new Date(rawValue);
    if (!Number.isNaN(parsed.valueOf())) {
      return parsed.toISOString().slice(0, 10);
    }
  }

  return undefined;
}

export function parseMarkdown(relativePath: string): ParsedMarkdown {
  if (useContentCache) {
    const cached = markdownCache.get(relativePath);
    if (cached) {
      return cached;
    }
  }

  const source = readFile(relativePath);
  const parsed = matter(source);
  const rawContent = parsed.content.replace(/\r\n?/g, "\n");
  assertSupportedMarkdown(rawContent, relativePath);
  const normalized = normalizeMarkdown(parsed.content);
  const { titleFromHeading, body } = removeLeadingHeading(normalized);

  const title =
    (typeof parsed.data.title === "string" ? collapseWhitespace(parsed.data.title) : undefined) ??
    titleFromHeading ??
    path.posix.basename(relativePath, path.posix.extname(relativePath));

  const excerptSource = rawContent.includes("<!-- more -->")
    ? rawContent.split("<!-- more -->")[0]
    : body;
  const excerpt = makeExcerpt(excerptSource);
  const description =
    ((typeof parsed.data.description === "string"
      ? collapseWhitespace(stripInlineMarkdown(parsed.data.description))
      : undefined) ??
      excerpt) ||
    title;

  const value: ParsedMarkdown = {
    title,
    description,
    excerpt,
    body,
    date: parseDate(parsed.data.date)
  };

  if (useContentCache) {
    markdownCache.set(relativePath, value);
  }
  return value;
}
