import Link from "next/link";
import { Fragment } from "react";

import type { SearchResult } from "../lib/content/types";
import { absoluteUrl, canonicalAlternates } from "../lib/seo";
import type { Locale } from "../lib/site-data";
import { getSearchResultsCopy } from "../lib/ui-copy";
import { SeoHead } from "./SeoHead";
import { SiteChrome } from "./SiteChrome";

type SearchResultsProps = {
  locale: Locale;
  query: string;
  results: SearchResult[];
};

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightMatches(value: string, query: string) {
  const terms = [...new Set(query.trim().split(/\s+/).filter((term) => term.length >= 2))];
  if (!terms.length) {
    return value;
  }

  const pattern = new RegExp(`(${terms.map(escapeRegExp).join("|")})`, "gi");

  return value.split(pattern).map((part, index) =>
    index % 2 === 1 ? (
      <mark key={`${part}-${index}`} className="rounded bg-mist px-0.5 text-ink">
        {part}
      </mark>
    ) : (
      <Fragment key={`${part}-${index}`}>{part}</Fragment>
    )
  );
}

export function SearchResults({ locale, query, results }: SearchResultsProps) {
  const copy = getSearchResultsCopy(locale, query, results.length);
  const path = locale === "zh" ? "/zh/search/" : "/search/";
  const shareHref = `https://x.com/intent/tweet?text=${encodeURIComponent(copy.title)}&url=${encodeURIComponent(
    absoluteUrl(`${path}?q=${encodeURIComponent(query.trim())}`)
  )}`;

  return (
    <>
      <SeoHead
        title={copy.title}
        description={copy.intro}
        path={path}
        alternates={canonicalAlternates({ en: "/search/", zh: "/zh/search/" })}
        eyebrow={copy.eyebrow}
        robots="noindex,follow"
      />
      <SiteChrome
        locale={locale}
        eyebrow={copy.eyebrow}
        title={copy.title}
        intro={copy.intro}
        alternates={{ en: "/search/", zh: "/zh/search/" }}
      >
        <section className="mx-auto max-w-5xl px-5 pb-16">
          {!query.trim() || query.trim().length < 2 ? (
            <div className="rounded-[2rem] border border-slate-200 bg-white/90 p-8 text-slate-600 shadow-panel">
              {copy.prompt}
            </div>
          ) : results.length ? (
            <div className="grid gap-4">
              <div className="flex justify-end">
                <a
                  href={shareHref}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-azure hover:text-azure"
                >
                  {copy.share}
                </a>
              </div>
              {results.map((result) => (
                <Link
                  key={`${result.locale}:${result.href}`}
                  href={result.href}
                  className="rounded-[1.75rem] border border-slate-200 bg-white/90 p-6 shadow-panel transition hover:border-azure hover:-translate-y-0.5"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-xl font-semibold tracking-tight text-ink">
                        {highlightMatches(result.title, query)}
                      </p>
                      <p className="mt-3 leading-7 text-slate-600">
                        {highlightMatches(result.description, query)}
                      </p>
                    </div>
                    <span className="rounded-full bg-mist px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-azure">
                      {result.section ?? result.kind}
                    </span>
                  </div>
                  <span className="mt-5 inline-flex text-sm font-semibold text-azure">{copy.open}</span>
                </Link>
              ))}
            </div>
          ) : (
            <div className="rounded-[2rem] border border-slate-200 bg-white/90 p-8 text-slate-600 shadow-panel">
              {copy.empty}
            </div>
          )}
        </section>
      </SiteChrome>
    </>
  );
}
