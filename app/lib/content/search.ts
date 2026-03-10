import fs from "node:fs";
import path from "node:path";

import type { Locale } from "../site-data";
import { useContentCache } from "./cache";
import { getDocument } from "./documents";
import { getContentManifest } from "./manifest";
import { markdownToSearchText } from "./markdown";
import { appRoot, generatedSearchDir } from "./roots";
import type { SearchResult } from "./types";

export type SearchDocument = SearchResult & {
  titleText: string;
  descriptionText: string;
  bodyTerms: string;
};

type SerializedSearchIndex = {
  locale: Locale;
  generatedAt: string;
  documents: SearchDocument[];
};

const SEARCH_SCORING = {
  TITLE_EXACT_MATCH: 500,
  TITLE_PREFIX_MATCH: 300,
  TITLE_CONTAINS: 180,
  DESCRIPTION_CONTAINS: 90,
  HREF_CONTAINS: 50,
  TOKEN_IN_TITLE: 60,
  TOKEN_IN_DESCRIPTION: 25,
  TOKEN_IN_BODY: 10,
} as const;

const MAX_BODY_TERMS = 192;

const supportedLocales: Locale[] = ["en", "zh"];
const searchIndexCache = new Map<Locale, SearchDocument[]>();
const publicSearchDir = path.join(appRoot, "public", "search-index");

function normalizeSearchValue(value: string): string {
  return value.toLowerCase().replace(/\s+/g, " ").trim();
}

function searchIndexPath(locale: Locale, outputDir: string = generatedSearchDir): string {
  return path.join(outputDir, `${locale}.json`);
}

function buildBodyTerms(body: string, headings: Array<{ text: string }>): string {
  const text = normalizeSearchValue(`${markdownToSearchText(body)} ${headings.map((heading) => heading.text).join(" ")}`);
  if (!text) {
    return "";
  }

  const tokens = text
    .split(" ")
    .filter((token) => token.length >= 3 || /\d/.test(token));

  const unique = new Set<string>();
  const limited: string[] = [];

  for (const token of tokens) {
    if (unique.has(token)) {
      continue;
    }

    unique.add(token);
    limited.push(token);

    if (limited.length >= MAX_BODY_TERMS) {
      break;
    }
  }

  return limited.join(" ");
}

function buildSearchDocumentsFromContent(locale: Locale): SearchDocument[] {
  const documents: SearchDocument[] = [];

  for (const record of getContentManifest()) {
    const href = record.routeByLocale[locale];
    const source = record.sourceByLocale[locale] ?? record.sourceByLocale.en ?? record.sourceByLocale.zh;
    if (!href || !source) {
      continue;
    }

    const document = getDocument(source);
    documents.push({
      title: document.title,
      description: document.description,
      href,
      locale,
      kind: record.kind,
      section: record.section,
      titleText: normalizeSearchValue(document.title),
      descriptionText: normalizeSearchValue(`${document.description} ${document.excerpt}`),
      bodyTerms: buildBodyTerms(document.body, document.headings)
    });

    for (const heading of document.headings) {
      documents.push({
        title: `${document.title} / ${heading.text}`,
        description: document.description || heading.text,
        href: `${href}#${heading.id}`,
        locale,
        kind: record.kind,
        section: record.section,
        titleText: normalizeSearchValue(`${heading.text} ${document.title}`),
        descriptionText: normalizeSearchValue(`${document.description} ${heading.text}`),
        bodyTerms: ""
      });
    }
  }

  return documents;
}

function allowSearchArtifactFallback(): boolean {
  return process.env.NODE_ENV === "development";
}

export function loadSearchDocuments(
  locale: Locale,
  options: {
    allowFallback?: boolean;
    outputDir?: string;
  } = {}
): SearchDocument[] {
  const outputDir = options.outputDir ?? generatedSearchDir;
  const fallbackAllowed = options.allowFallback ?? allowSearchArtifactFallback();

  if (useContentCache && outputDir === generatedSearchDir) {
    const cached = searchIndexCache.get(locale);
    if (cached) {
      return cached;
    }
  }

  const documents = readPrebuiltSearchDocumentsFrom(locale, outputDir);
  if (documents) {
    if (useContentCache && outputDir === generatedSearchDir) {
      searchIndexCache.set(locale, documents);
    }
    return documents;
  }

  if (!fallbackAllowed) {
    throw new Error(
      `Missing prebuilt search index for ${locale} at ${searchIndexPath(locale, outputDir)}. Run generate:search-index first.`
    );
  }

  const rebuilt = buildSearchDocumentsFromContent(locale);
  if (useContentCache && outputDir === generatedSearchDir) {
    searchIndexCache.set(locale, rebuilt);
  }
  return rebuilt;
}

function readPrebuiltSearchDocumentsFrom(
  locale: Locale,
  outputDir: string
): SearchDocument[] | null {
  const filePath = searchIndexPath(locale, outputDir);
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
    if (!allowSearchArtifactFallback() && outputDir === generatedSearchDir) {
      throw new Error(`Failed to read prebuilt search index for ${locale}: ${String(error)}`);
    }

    console.warn(`Failed to read prebuilt search index for ${locale}. Falling back to content scan.`, error);
    return null;
  }
}

function scoreDocument(document: SearchDocument, query: string, tokens: string[]): number {
  let score = 0;

  if (document.titleText === query) {
    score += SEARCH_SCORING.TITLE_EXACT_MATCH;
  } else if (document.titleText.startsWith(query)) {
    score += SEARCH_SCORING.TITLE_PREFIX_MATCH;
  } else if (document.titleText.includes(query)) {
    score += SEARCH_SCORING.TITLE_CONTAINS;
  }

  if (document.descriptionText.includes(query)) {
    score += SEARCH_SCORING.DESCRIPTION_CONTAINS;
  }

  if (document.href.toLowerCase().includes(query)) {
    score += SEARCH_SCORING.HREF_CONTAINS;
  }

  let matchedTokens = 0;
  for (const token of tokens) {
    if (document.titleText.includes(token)) {
      score += SEARCH_SCORING.TOKEN_IN_TITLE;
      matchedTokens += 1;
      continue;
    }

    if (document.descriptionText.includes(token)) {
      score += SEARCH_SCORING.TOKEN_IN_DESCRIPTION;
      matchedTokens += 1;
      continue;
    }

    if (document.bodyTerms.includes(token)) {
      score += SEARCH_SCORING.TOKEN_IN_BODY;
      matchedTokens += 1;
    }
  }

  if (matchedTokens !== tokens.length) {
    return 0;
  }

  return score;
}

export function searchDocuments(
  documents: SearchDocument[],
  query: string,
  limit: number = 8
): SearchResult[] {
  const normalizedQuery = normalizeSearchValue(query);
  if (normalizedQuery.length < 2) {
    return [];
  }

  const tokens = normalizedQuery.split(" ").filter(Boolean);

  return documents
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

function writeSearchIndexPayload(
  locale: Locale,
  documents: SearchDocument[],
  outputDir: string
) {
  fs.mkdirSync(outputDir, { recursive: true });
  const filePath = searchIndexPath(locale, outputDir);
  const tempPath = `${filePath}.tmp`;
  const payload: SerializedSearchIndex = {
    locale,
    generatedAt: new Date().toISOString(),
    documents
  };

  fs.writeFileSync(tempPath, `${JSON.stringify(payload)}\n`, "utf8");
  fs.renameSync(tempPath, filePath);

  return filePath;
}

export function writeSearchIndexes(outputDir: string = generatedSearchDir) {
  fs.mkdirSync(outputDir, { recursive: true });
  fs.mkdirSync(publicSearchDir, { recursive: true });

  return supportedLocales.map((locale) => {
    const documents = buildSearchDocumentsFromContent(locale);
    const filePath = writeSearchIndexPayload(locale, documents, outputDir);
    writeSearchIndexPayload(locale, documents, publicSearchDir);
    searchIndexCache.set(locale, documents);

    return {
      locale,
      count: documents.length,
      filePath
    };
  });
}

export function searchContent(query: string, locale: Locale, limit: number = 8): SearchResult[] {
  return searchDocuments(getSearchDocuments(locale), query, limit);
}

function getSearchDocuments(locale: Locale): SearchDocument[] {
  return loadSearchDocuments(locale);
}
