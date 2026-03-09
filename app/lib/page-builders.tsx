import { useRouter } from "next/router";
import { useDeferredValue, useEffect, useState } from "react";

import { SearchResults } from "../components/SearchResults";
import type { DocsPage, SearchResult, SidebarGroup } from "./content/types";
import { DocsPageView, HomePageView, type HomePageData } from "./page-factories";
import type { Locale } from "./site-data";

type SearchPageProps = {
  sidebar: SidebarGroup[];
};

type StaticSearchDocument = SearchResult & {
  titleText: string;
  descriptionText: string;
  bodyTerms: string;
};

function normalizeSearchValue(value: string): string {
  return value.toLowerCase().replace(/\s+/g, " ").trim();
}

function scoreDocument(document: StaticSearchDocument, query: string, tokens: string[]): number {
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

    if (document.bodyTerms.includes(token)) {
      score += 10;
      matchedTokens += 1;
    }
  }

  if (matchedTokens !== tokens.length) {
    return 0;
  }

  return score;
}

function searchDocuments(
  documents: StaticSearchDocument[],
  query: string,
  limit: number = 24
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

type StaticSearchIndex = {
  documents: StaticSearchDocument[];
};

const searchIndexCache = new Map<Locale, Promise<StaticSearchDocument[]>>();

function loadSearchDocuments(locale: Locale): Promise<StaticSearchDocument[]> {
  const cached = searchIndexCache.get(locale);
  if (cached) {
    return cached;
  }

  const pending = fetch(`/search-index/${locale}.json`)
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`Failed to load static search index for ${locale}: ${response.status}`);
      }

      const payload = (await response.json()) as StaticSearchIndex;
      if (!Array.isArray(payload.documents)) {
        throw new Error(`Invalid static search index payload for ${locale}`);
      }

      return payload.documents;
    });

  searchIndexCache.set(locale, pending);
  return pending;
}

export type ContentPageProps = {
  page: DocsPage;
  eyebrow: string;
};

export function createHomePage(locale: Locale, eyebrow: string) {
  return function HomePage({ page }: { page: HomePageData }) {
    return <HomePageView page={page} locale={locale} eyebrow={eyebrow} />;
  };
}

export function createSearchPage(locale: Locale) {
  return function SearchPage({ sidebar }: SearchPageProps) {
    const router = useRouter();
    const rawQuery = typeof router.query.q === "string" ? router.query.q : "";
    const deferredQuery = useDeferredValue(rawQuery);
    const [results, setResults] = useState<SearchResult[]>([]);

    useEffect(() => {
      let cancelled = false;
      const query = deferredQuery.trim();

      if (query.length < 2) {
        setResults([]);
        return () => {
          cancelled = true;
        };
      }

      loadSearchDocuments(locale)
        .then((documents) => {
          if (cancelled) {
            return;
          }

          setResults(searchDocuments(documents, query, 24));
        })
        .catch(() => {
          if (!cancelled) {
            setResults([]);
          }
        });

      return () => {
        cancelled = true;
      };
    }, [deferredQuery]);

    return <SearchResults locale={locale} query={rawQuery} results={results} sidebar={sidebar} />;
  };
}

export function createContentPage(locale: Locale) {
  return function ContentPage({ page, eyebrow }: ContentPageProps) {
    return <DocsPageView page={page} locale={locale} eyebrow={eyebrow} />;
  };
}

export function createFeedPage() {
  return function FeedPage() {
    return null;
  };
}
