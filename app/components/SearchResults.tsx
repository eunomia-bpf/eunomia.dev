import Link from "next/link";
import { Fragment } from "react";

import type { SearchResult, SidebarGroup } from "../lib/content/types";
import { localizePath } from "../lib/paths";
import { absoluteUrl, canonicalAlternates } from "../lib/seo";
import type { Locale } from "../lib/site-data";
import { getSearchResultsCopy } from "../lib/ui-copy";
import { SeoHead } from "./SeoHead";
import { SiteChrome } from "./SiteChrome";

type SearchResultsProps = {
  locale: Locale;
  query: string;
  results: SearchResult[];
  sidebar: SidebarGroup[];
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
      <mark key={`${part}-${index}`} className="rounded bg-slate-200 px-0.5 text-ink">
        {part}
      </mark>
    ) : (
      <Fragment key={`${part}-${index}`}>{part}</Fragment>
    )
  );
}

export function SearchResults({ locale, query, results, sidebar }: SearchResultsProps) {
  const copy = getSearchResultsCopy(locale, query, results.length);
  const path = localizePath("/search/", locale);
  const shareHref = `https://x.com/intent/tweet?text=${encodeURIComponent(copy.title)}&url=${encodeURIComponent(
    absoluteUrl(`${path}?q=${encodeURIComponent(query.trim())}`)
  )}`;

  return (
    <>
      <SeoHead
        title={copy.title}
        description={copy.intro}
        path={path}
        alternates={canonicalAlternates({ en: localizePath("/search/", "en"), zh: localizePath("/search/", "zh") })}
        eyebrow={copy.eyebrow}
        robots="noindex,follow"
      />
      <SiteChrome
        locale={locale}
        eyebrow={copy.eyebrow}
        title={copy.title}
        intro={copy.intro}
        currentPath={path}
        sidebar={sidebar}
        alternates={{ en: localizePath("/search/", "en"), zh: localizePath("/search/", "zh") }}
      >
        <section className="pb-16">
          {!query.trim() || query.trim().length < 2 ? (
            <div className="rounded-xl border border-slate-200 bg-white p-8 text-slate-600">
              {copy.prompt}
            </div>
          ) : results.length ? (
            <div className="space-y-4">
              <div className="flex justify-end">
                <a
                  href={shareHref}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex rounded-lg border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 hover:text-ink"
                >
                  {copy.share}
                </a>
              </div>
              <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
                {results.map((result, index) => (
                  <Link
                    key={`${result.locale}:${result.href}`}
                    href={result.href}
                    className={`block px-6 py-5 transition hover:bg-slate-50 ${
                      index > 0 ? "border-t border-slate-200" : ""
                    }`}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0">
                        <p className="text-lg font-semibold tracking-tight text-ink">
                          {highlightMatches(result.title, query)}
                        </p>
                        <p className="mt-2 leading-7 text-slate-600">
                          {highlightMatches(result.description, query)}
                        </p>
                        <span className="mt-4 inline-flex text-sm font-semibold text-slate-900">{copy.open}</span>
                      </div>
                      <span className="rounded-md bg-slate-100 px-2.5 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-slate-600">
                        {result.section ?? result.kind}
                      </span>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-slate-200 bg-white p-8 text-slate-600">
              {copy.empty}
            </div>
          )}
        </section>
      </SiteChrome>
    </>
  );
}
