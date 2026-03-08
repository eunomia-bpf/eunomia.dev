"use client";

import Link from "next/link";
import { useRouter } from "next/router";
import { startTransition, useDeferredValue, useEffect, useId, useRef, useState } from "react";

import type { Locale } from "../lib/site-data";
import type { SearchResult } from "../lib/content/types";
import { localizePath } from "../lib/paths";
import { searchBoxCopyByLocale } from "../lib/ui-copy";

type SearchBoxProps = {
  locale: Locale;
  containerClassName?: string;
  inputClassName?: string;
  panelClassName?: string;
  onNavigate?: () => void;
};

type SearchResponse = {
  documents: StaticSearchDocument[];
};

type StaticSearchDocument = SearchResult & {
  titleText: string;
  descriptionText: string;
  bodyTerms: string;
};

const searchIndexCache = new Map<Locale, Promise<StaticSearchDocument[]>>();

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

function loadSearchDocuments(locale: Locale): Promise<StaticSearchDocument[]> {
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

export function SearchBox({
  locale,
  containerClassName = "",
  inputClassName = "",
  panelClassName = "",
  onNavigate
}: SearchBoxProps) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [requestFailed, setRequestFailed] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const deferredQuery = useDeferredValue(query);
  const requestRef = useRef<AbortController | null>(null);
  const listboxId = useId();
  const labels = searchBoxCopyByLocale[locale];
  const normalizedQuery = deferredQuery.trim();
  const hasQuery = normalizedQuery.length >= 2;
  const searchHref = `${localizePath("/search/", locale)}?q=${encodeURIComponent(query.trim())}`;

  useEffect(() => {
    requestRef.current?.abort();
    setActiveIndex(-1);

    if (!hasQuery) {
      startTransition(() => setResults([]));
      setRequestFailed(false);
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    requestRef.current = controller;
    setRequestFailed(false);
    setLoading(true);

    loadSearchDocuments(locale)
      .then((documents) => {
        if (controller.signal.aborted) {
          return;
        }

        startTransition(() => setResults(searchDocuments(documents, normalizedQuery, 8)));
      })
      .catch((error: unknown) => {
        if ((error as { name?: string })?.name !== "AbortError") {
          startTransition(() => setResults([]));
          setRequestFailed(true);
        }
      })
      .finally(() => {
        if (requestRef.current === controller) {
          setLoading(false);
        }
      });

    return () => {
      controller.abort();
    };
  }, [hasQuery, normalizedQuery, locale]);

  const showResults = open && (hasQuery || loading);

  function navigateTo(href: string) {
    onNavigate?.();
    setOpen(false);
    void router.push(href);
  }

  return (
    <div className={`relative ${containerClassName}`.trim()}>
      <label>
        <span className="sr-only">{labels.aria}</span>
        <input
          aria-label={labels.aria}
          aria-expanded={showResults}
          aria-controls={listboxId}
          aria-activedescendant={activeIndex >= 0 ? `${listboxId}-option-${activeIndex}` : undefined}
          aria-autocomplete="list"
          aria-haspopup="listbox"
          role="combobox"
          type="text"
          value={query}
          placeholder={labels.placeholder}
          className={`rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm outline-none transition focus:border-slate-400 ${inputClassName}`.trim()}
          onFocus={() => setOpen(true)}
          onBlur={() => {
            window.setTimeout(() => setOpen(false), 120);
          }}
          onKeyDown={(event) => {
            if (event.key === "ArrowDown") {
              event.preventDefault();
              setOpen(true);
              if (results.length) {
                setActiveIndex((index) => (index + 1 + results.length) % results.length);
              }
              return;
            }

            if (event.key === "ArrowUp") {
              event.preventDefault();
              setOpen(true);
              if (results.length) {
                setActiveIndex((index) => (index <= 0 ? results.length - 1 : index - 1));
              }
              return;
            }

            if (event.key === "Enter" && hasQuery) {
              event.preventDefault();
              if (activeIndex >= 0 && results[activeIndex]) {
                navigateTo(results[activeIndex].href);
                return;
              }

              navigateTo(searchHref);
              return;
            }

            if (event.key === "Escape") {
              setOpen(false);
              setActiveIndex(-1);
            }
          }}
          onChange={(event) => {
            setQuery(event.target.value);
            setOpen(true);
          }}
        />
      </label>
      {showResults ? (
        <div
          id={listboxId}
          className={`absolute right-0 top-[calc(100%+0.75rem)] z-50 w-[28rem] overflow-hidden rounded-xl border border-slate-200 bg-white ${panelClassName}`.trim()}
          onMouseDown={(event) => {
            event.preventDefault();
          }}
        >
          <div className="border-b border-slate-100 px-4 py-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
            {loading ? labels.loading : requestFailed ? labels.error : labels.results}
          </div>
          {results.length ? (
            <ul role="listbox" className="max-h-[28rem] overflow-y-auto">
              {results.map((result, index) => (
                <li key={`${result.locale}:${result.href}`} className="border-t border-slate-100 first:border-t-0">
                  <Link
                    href={result.href}
                    id={`${listboxId}-option-${index}`}
                    role="option"
                    aria-selected={activeIndex === index}
                    className={`block px-4 py-3 transition hover:bg-slate-50 ${
                      activeIndex === index ? "bg-slate-50" : ""
                    }`}
                    onMouseEnter={() => setActiveIndex(index)}
                    onClick={() => onNavigate?.()}
                  >
                    <p className="text-sm font-semibold text-ink">{result.title}</p>
                    <p className="mt-1 line-clamp-2 text-sm text-slate-600">{result.description}</p>
                    <p className="mt-2 text-xs uppercase tracking-[0.16em] text-slate-400">
                      {result.section ?? result.kind}
                    </p>
                  </Link>
                </li>
              ))}
            </ul>
          ) : hasQuery && !loading ? (
            <div className="px-4 py-5 text-sm text-slate-500">{requestFailed ? labels.error : labels.empty}</div>
          ) : null}
          {hasQuery ? (
            <div className="border-t border-slate-100 px-4 py-3">
              <Link
                href={searchHref}
                className="text-sm font-semibold text-ink underline decoration-slate-300 underline-offset-4 transition hover:decoration-slate-500"
                onClick={() => onNavigate?.()}
              >
                {labels.viewAll}
              </Link>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
