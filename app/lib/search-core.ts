import type { SearchResult } from "./content/types";

export type SearchDocument = SearchResult & {
  titleText: string;
  descriptionText: string;
  bodyTerms: string;
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

export function normalizeSearchValue(value: string): string {
  return value.toLowerCase().replace(/\s+/g, " ").trim();
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

  return matchedTokens === tokens.length ? score : 0;
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
