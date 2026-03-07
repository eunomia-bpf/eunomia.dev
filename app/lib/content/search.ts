import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import type { Locale } from "../site-data";
import { useContentCache } from "./cache";
import { getContentManifest } from "./manifest";
import { extractMarkdownHeadings, markdownToSearchText, parseMarkdown } from "./markdown";
import type { SearchResult } from "./types";

type SearchDocument = SearchResult & {
  titleText: string;
  descriptionText: string;
  bodyText: string;
};

type SerializedSearchIndex = {
  locale: Locale;
  generatedAt: string;
  documents: SearchDocument[];
};

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const generatedSearchDir = path.join(appRoot, ".generated", "search");
const supportedLocales: Locale[] = ["en", "zh"];
const searchIndexCache = new Map<Locale, SearchDocument[]>();

function normalizeSearchValue(value: string): string {
  return value.toLowerCase().replace(/\s+/g, " ").trim();
}

function searchIndexPath(locale: Locale, outputDir: string = generatedSearchDir): string {
  return path.join(outputDir, `${locale}.json`);
}

function buildSearchDocumentsFromContent(locale: Locale): SearchDocument[] {
  const documents: SearchDocument[] = [];

  for (const record of getContentManifest()) {
    const href = record.routeByLocale[locale];
    const source = record.sourceByLocale[locale] ?? record.sourceByLocale.en ?? record.sourceByLocale.zh;
    if (!href || !source) {
      continue;
    }

    const parsed = parseMarkdown(source);
    documents.push({
      title: parsed.title,
      description: parsed.description,
      href,
      locale,
      kind: record.kind,
      section: record.section,
      titleText: normalizeSearchValue(parsed.title),
      descriptionText: normalizeSearchValue(`${parsed.description} ${parsed.excerpt}`),
      bodyText: normalizeSearchValue(markdownToSearchText(parsed.body))
    });

    for (const heading of extractMarkdownHeadings(parsed.body)) {
      documents.push({
        title: `${parsed.title} / ${heading.text}`,
        description: parsed.description || heading.text,
        href: `${href}#${heading.id}`,
        locale,
        kind: record.kind,
        section: record.section,
        titleText: normalizeSearchValue(`${heading.text} ${parsed.title}`),
        descriptionText: normalizeSearchValue(`${parsed.description} ${heading.text}`),
        bodyText: ""
      });
    }
  }

  return documents;
}

function readPrebuiltSearchDocuments(locale: Locale): SearchDocument[] | null {
  const filePath = searchIndexPath(locale);
  if (!fs.existsSync(filePath)) {
    return null;
  }

  try {
    const payload = JSON.parse(fs.readFileSync(filePath, "utf8")) as SerializedSearchIndex;
    if (!Array.isArray(payload.documents)) {
      return null;
    }

    return payload.documents;
  } catch (error) {
    console.warn(`Failed to read prebuilt search index for ${locale}. Falling back to content scan.`, error);
    return null;
  }
}

function getSearchDocuments(locale: Locale): SearchDocument[] {
  if (useContentCache) {
    const cached = searchIndexCache.get(locale);
    if (cached) {
      return cached;
    }
  }

  const documents = readPrebuiltSearchDocuments(locale) ?? buildSearchDocumentsFromContent(locale);
  if (useContentCache) {
    searchIndexCache.set(locale, documents);
  }
  return documents;
}

function scoreDocument(document: SearchDocument, query: string, tokens: string[]): number {
  let score = 0;

  if (document.titleText === query) {
    score += 500;
  } else if (document.titleText.startsWith(query)) {
    score += 300;
  } else if (document.titleText.includes(query)) {
    score += 180;
  }

  if (document.descriptionText.includes(query)) {
    score += 90;
  }

  if (document.href.toLowerCase().includes(query)) {
    score += 50;
  }

  let matchedTokens = 0;
  for (const token of tokens) {
    if (document.titleText.includes(token)) {
      score += 60;
      matchedTokens += 1;
      continue;
    }

    if (document.descriptionText.includes(token)) {
      score += 25;
      matchedTokens += 1;
      continue;
    }

    if (document.bodyText.includes(token)) {
      score += 10;
      matchedTokens += 1;
    }
  }

  if (matchedTokens !== tokens.length) {
    return 0;
  }

  return score;
}

export function writeSearchIndexes(outputDir: string = generatedSearchDir) {
  fs.mkdirSync(outputDir, { recursive: true });

  return supportedLocales.map((locale) => {
    const documents = buildSearchDocumentsFromContent(locale);
    const payload: SerializedSearchIndex = {
      locale,
      generatedAt: new Date().toISOString(),
      documents
    };

    fs.writeFileSync(searchIndexPath(locale, outputDir), `${JSON.stringify(payload, null, 2)}\n`, "utf8");
    searchIndexCache.set(locale, documents);

    return {
      locale,
      count: documents.length,
      filePath: searchIndexPath(locale, outputDir)
    };
  });
}

export function searchContent(query: string, locale: Locale, limit: number = 8): SearchResult[] {
  const normalizedQuery = normalizeSearchValue(query);
  if (normalizedQuery.length < 2) {
    return [];
  }

  const tokens = normalizedQuery.split(" ").filter(Boolean);

  return getSearchDocuments(locale)
    .map((document) => ({
      document,
      score: scoreDocument(document, normalizedQuery, tokens)
    }))
    .filter((entry) => entry.score > 0)
    .sort((left, right) => {
      if (right.score !== left.score) {
        return right.score - left.score;
      }

      return left.document.title.length - right.document.title.length;
    })
    .slice(0, limit)
    .map(({ document }) => ({
      title: document.title,
      description: document.description,
      href: document.href,
      locale: document.locale,
      kind: document.kind,
      section: document.section
    }));
}
