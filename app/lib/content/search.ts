import fs from "node:fs";
import path from "node:path";

import type { Locale } from "../site-data";
import { useContentCache } from "./cache";
import { getDocument } from "./documents";
import { getContentManifest } from "./manifest";
import { markdownToSearchText } from "./markdown";
import { appRoot, generatedSearchDir } from "./roots";
import type { SearchResult } from "./types";
import { normalizeSearchValue, searchDocuments, type SearchDocument } from "../search-core";

type SerializedSearchIndex = {
  locale: Locale;
  generatedAt: string;
  documents: SearchDocument[];
};

const MAX_BODY_TERMS = 128;
const MAX_HEADING_SEARCH_DEPTH = 2;

const supportedLocales: Locale[] = ["en", "zh"];
const searchIndexCache = new Map<Locale, SearchDocument[]>();
const publicSearchDir = path.join(appRoot, "public", "search-index");

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
    const descriptionText =
      document.description === document.excerpt
        ? document.description
        : `${document.description} ${document.excerpt}`;
    const tagText = document.tags.join(" ");

    documents.push({
      title: document.title,
      description: document.description,
      href,
      locale,
      kind: record.kind,
      section: record.section,
      titleText: normalizeSearchValue(document.title),
      descriptionText: normalizeSearchValue(`${descriptionText} ${tagText}`),
      bodyTerms: buildBodyTerms(`${tagText}\n\n${document.body}`, document.headings)
    });

    for (const heading of document.headings) {
      if (heading.depth > MAX_HEADING_SEARCH_DEPTH) {
        continue;
      }

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
