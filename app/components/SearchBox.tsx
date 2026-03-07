import { startTransition, useDeferredValue, useEffect, useId, useRef, useState } from "react";

import type { Locale } from "../lib/site-data";
import type { SearchResult } from "../lib/content/types";

type SearchBoxProps = {
  locale: Locale;
};

type SearchResponse = {
  results: SearchResult[];
};

export function SearchBox({ locale }: SearchBoxProps) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const deferredQuery = useDeferredValue(query);
  const requestRef = useRef<AbortController | null>(null);
  const listboxId = useId();

  useEffect(() => {
    const normalized = deferredQuery.trim();
    requestRef.current?.abort();

    if (normalized.length < 2) {
      startTransition(() => setResults([]));
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    requestRef.current = controller;
    setLoading(true);

    fetch(`/api/search?q=${encodeURIComponent(normalized)}&locale=${locale}`, {
      signal: controller.signal
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Search request failed with ${response.status}`);
        }

        const payload = (await response.json()) as SearchResponse;
        startTransition(() => setResults(payload.results));
      })
      .catch((error: unknown) => {
        if ((error as { name?: string })?.name !== "AbortError") {
          startTransition(() => setResults([]));
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
  }, [deferredQuery, locale]);

  const hasQuery = deferredQuery.trim().length >= 2;
  const showResults = open && (hasQuery || loading);

  return (
    <div className="relative hidden md:block">
      <label>
        <span className="sr-only">{locale === "zh" ? "搜索" : "Search"}</span>
        <input
          aria-label={locale === "zh" ? "Search" : "Search"}
          aria-expanded={showResults}
          aria-controls={listboxId}
          type="text"
          value={query}
          placeholder={locale === "zh" ? "搜索文档" : "Search docs"}
          className="w-52 rounded-full border border-slate-200 bg-slate-50 px-4 py-2 text-sm outline-none transition focus:border-azure"
          onFocus={() => setOpen(true)}
          onBlur={() => {
            window.setTimeout(() => setOpen(false), 120);
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
          className="absolute right-0 top-[calc(100%+0.75rem)] z-50 w-[28rem] overflow-hidden rounded-[1.5rem] border border-slate-200 bg-white shadow-2xl"
        >
          <div className="border-b border-slate-100 px-4 py-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
            {loading
              ? locale === "zh"
                ? "搜索中"
                : "Searching"
              : locale === "zh"
                ? "搜索结果"
                : "Results"}
          </div>
          {results.length ? (
            <ul className="max-h-[28rem] overflow-y-auto">
              {results.map((result) => (
                <li key={`${result.locale}:${result.href}`} className="border-t border-slate-100 first:border-t-0">
                  <a
                    href={result.href}
                    className="block px-4 py-3 transition hover:bg-slate-50"
                  >
                    <p className="text-sm font-semibold text-ink">{result.title}</p>
                    <p className="mt-1 line-clamp-2 text-sm text-slate-600">{result.description}</p>
                    <p className="mt-2 text-xs uppercase tracking-[0.16em] text-slate-400">
                      {result.section ?? result.kind}
                    </p>
                  </a>
                </li>
              ))}
            </ul>
          ) : hasQuery && !loading ? (
            <div className="px-4 py-5 text-sm text-slate-500">
              {locale === "zh" ? "没有找到匹配结果。" : "No matching results."}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
