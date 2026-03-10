import type { SearchResult } from "./content/types";
import type { Locale } from "./site-data";

export type StaticSearchDocument = SearchResult & {
  titleText: string;
  descriptionText: string;
  bodyTerms: string;
};

type SearchResponse = {
  documents: StaticSearchDocument[];
};

const searchIndexCache = new Map<Locale, Promise<StaticSearchDocument[]>>();

export function normalizeSearchValue(value: string): string {
  return value.toLowerCase().replace(/\s+/g, " ").trim();
}

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

export function scoreDocument(document: StaticSearchDocument, query: string, tokens: string[]): number {
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
  documents: StaticSearchDocument[],
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

export function loadSearchDocuments(locale: Locale): Promise<StaticSearchDocument[]> {
  const cached = searchIndexCache.get(locale);
  if (cached) {
    return cached;
  }

  const pending = fetch(`/search-index/${locale}.json`).then(async (response) => {
    if (!response.ok) {
      throw new Error(`Failed to load static search index for ${locale}: ${response.status}`);
    }

    const payload = (await response.json()) as SearchResponse;
    if (!Array.isArray(payload.documents)) {
      throw new Error(`Invalid static search payload for ${locale}`);
    }

    return payload.documents;
  });

  searchIndexCache.set(locale, pending);
  return pending;
}
