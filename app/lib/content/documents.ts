import fs from "node:fs";
import path from "node:path";

import { useContentCache } from "./cache";
import { getDocsFileSet } from "./fs-index";
import { extractMarkdownHeadings, parseMarkdown, type MarkdownHeading } from "./markdown";
import { readMkdocsSectionPage } from "./mkdocs-config";
import { generatedContentDir } from "./roots";
import { resolveLocalizedSource } from "./source";

export type IndexedDocument = {
  sourceRelative: string;
  title: string;
  description: string;
  excerpt: string;
  body: string;
  date?: string;
  slug?: string;
  headings: MarkdownHeading[];
};

type SerializedDocumentIndex = {
  generatedAt: string;
  documents: IndexedDocument[];
};

const generatedDocumentIndexPath = path.join(generatedContentDir, "documents.json");
const documentIndexCache = new Map<string, IndexedDocument>();

function localeFromSourceRelative(sourceRelative: string): "en" | "zh" {
  return /\.(zh|zh-CN)\.md$/.test(sourceRelative) ? "zh" : "en";
}

function buildDocument(sourceRelative: string): IndexedDocument {
  const parsed = parseMarkdown(sourceRelative);
  const configuredPage = readMkdocsSectionPage(sourceRelative);
  const locale = localeFromSourceRelative(sourceRelative);
  const title = configuredPage?.title[locale] ?? parsed.title;
  const description = configuredPage?.description[locale] ?? parsed.description;
  const body = configuredPage?.body[locale] ?? parsed.body;

  return {
    sourceRelative,
    title,
    description,
    excerpt: configuredPage ? description : parsed.excerpt,
    body,
    date: parsed.date,
    slug: parsed.slug,
    headings: extractMarkdownHeadings(body)
  };
}

function buildDocumentIndexFromSource(): IndexedDocument[] {
  return [...getDocsFileSet()]
    .filter((relativePath) => relativePath.endsWith(".md"))
    .sort((left, right) =>
      left.localeCompare(right, "en", {
        numeric: true,
        sensitivity: "base"
      })
    )
    .map((relativePath) => buildDocument(relativePath));
}

function hydrateDocumentCache(documents: IndexedDocument[]) {
  documentIndexCache.clear();
  for (const document of documents) {
    documentIndexCache.set(document.sourceRelative, document);
  }
}

function readPrebuiltDocumentIndex(): IndexedDocument[] | null {
  if (!fs.existsSync(generatedDocumentIndexPath)) {
    return null;
  }

  try {
    const payload = JSON.parse(fs.readFileSync(generatedDocumentIndexPath, "utf8")) as SerializedDocumentIndex;
    if (!Array.isArray(payload.documents)) {
      return null;
    }

    return payload.documents;
  } catch (error) {
    console.warn("Failed to read prebuilt document index. Falling back to content scan.", error);
    return null;
  }
}

export function getDocumentIndex(): Map<string, IndexedDocument> {
  if (useContentCache && documentIndexCache.size) {
    return documentIndexCache;
  }

  const documents = readPrebuiltDocumentIndex() ?? buildDocumentIndexFromSource();
  hydrateDocumentCache(documents);

  return documentIndexCache;
}

export function listDocuments(): IndexedDocument[] {
  return [...getDocumentIndex().values()];
}

export function getDocument(sourceRelative: string): IndexedDocument {
  const cached = getDocumentIndex().get(sourceRelative);
  if (cached) {
    return cached;
  }

  const document = buildDocument(sourceRelative);
  if (useContentCache) {
    documentIndexCache.set(sourceRelative, document);
  }
  return document;
}

export function getDocumentBySource(sourceRelative: string): IndexedDocument | null {
  return getDocumentIndex().get(sourceRelative) ?? null;
}

export function resolveDocument(relativePath: string, locale: "en" | "zh"): IndexedDocument | null {
  const sourceRelative = resolveLocalizedSource(relativePath, locale);
  return sourceRelative ? getDocument(sourceRelative) : null;
}

export function writeDocumentIndex(outputPath: string = generatedDocumentIndexPath) {
  const documents = buildDocumentIndexFromSource();
  const payload: SerializedDocumentIndex = {
    generatedAt: new Date().toISOString(),
    documents
  };

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const tempPath = `${outputPath}.tmp`;
  fs.writeFileSync(tempPath, `${JSON.stringify(payload)}\n`, "utf8");
  fs.renameSync(tempPath, outputPath);
  hydrateDocumentCache(documents);

  return {
    count: documents.length,
    filePath: outputPath
  };
}
