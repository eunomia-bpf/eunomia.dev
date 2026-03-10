"use client";

import Link from "next/link";
import { useRouter } from "next/router";
import { startTransition, useDeferredValue, useEffect, useId, useRef, useState } from "react";

import type { Locale } from "../lib/site-data";
import type { SearchResult } from "../lib/content/types";
import { loadSearchDocuments, searchDocuments } from "../lib/client-search";
import { localizePath } from "../lib/paths";
import { searchBoxCopyByLocale } from "../lib/ui-copy";

type SearchBoxProps = {
  locale: Locale;
  containerClassName?: string;
  inputClassName?: string;
  panelClassName?: string;
  onNavigate?: () => void;
};

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
  const BLUR_DELAY_MS = 120; // Allow click events on dropdown items to register before closing
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
          className={`rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm outline-none transition focus:border-slate-400 focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:ring-offset-1 ${inputClassName}`.trim()}
          onFocus={() => setOpen(true)}
          onBlur={() => {
            window.setTimeout(() => setOpen(false), BLUR_DELAY_MS);
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
