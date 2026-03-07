import fs from "node:fs";
import path from "node:path";

import matter from "gray-matter";

import { docsRoot } from "./fs-index";
import type { ParsedMarkdown } from "./types";

const markdownCache = new Map<string, ParsedMarkdown>();

function readFile(relativePath: string, root: string = docsRoot): string {
  return fs.readFileSync(path.join(root, relativePath), "utf8");
}

function removeLeadingHeading(markdown: string): { titleFromHeading?: string; body: string } {
  const headingMatch = markdown.match(/^\s*#\s+(.+?)\s*$/m);
  if (!headingMatch) {
    return { body: markdown.trim() };
  }

  const titleFromHeading = stripInlineMarkdown(headingMatch[1]);
  const body = markdown.replace(/^\s*#\s+(.+?)\s*$(?:\n+)?/m, "").trim();
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
  const cached = markdownCache.get(relativePath);
  if (cached) {
    return cached;
  }

  const source = readFile(relativePath);
  const parsed = matter(source);
  const rawContent = parsed.content.replace(/\r\n?/g, "\n");
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

  markdownCache.set(relativePath, value);
  return value;
}
