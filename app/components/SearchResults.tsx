import type { SearchResult } from "../lib/content/types";
import { canonicalAlternates } from "../lib/seo";
import type { Locale } from "../lib/site-data";
import { SeoHead } from "./SeoHead";
import { SiteChrome } from "./SiteChrome";

type SearchResultsProps = {
  locale: Locale;
  query: string;
  results: SearchResult[];
};

export function SearchResults({ locale, query, results }: SearchResultsProps) {
  const copy =
    locale === "zh"
      ? {
          eyebrow: "搜索",
          title: query ? `“${query}” 的搜索结果` : "搜索站点内容",
          intro: query
            ? `找到 ${results.length} 条匹配结果。`
            : "输入至少 2 个字符，通过 header 搜索框检索教程、博客和文档页面。",
          empty: "没有找到匹配结果，可以换个关键词再试。",
          prompt: "输入至少 2 个字符后，再从 header 的搜索框发起搜索。",
          open: "打开结果"
        }
      : {
          eyebrow: "Search",
          title: query ? `Results for “${query}”` : "Search the site",
          intro: query
            ? `Found ${results.length} matching results.`
            : "Use at least 2 characters from the header search box to search tutorials, blog posts, and docs.",
          empty: "No matching results yet. Try a broader or more specific query.",
          prompt: "Use at least 2 characters, then search from the header search box.",
          open: "Open result"
        };
  const path = locale === "zh" ? "/zh/search/" : "/search/";

  return (
    <>
      <SeoHead
        title={copy.title}
        description={copy.intro}
        path={path}
        alternates={canonicalAlternates("/search/", "/zh/search/")}
        eyebrow={copy.eyebrow}
        robots="noindex,follow"
      />
      <SiteChrome locale={locale} eyebrow={copy.eyebrow} title={copy.title} intro={copy.intro}>
        <section className="mx-auto max-w-5xl px-5 pb-16">
          {!query.trim() || query.trim().length < 2 ? (
            <div className="rounded-[2rem] border border-slate-200 bg-white/90 p-8 text-slate-600 shadow-panel">
              {copy.prompt}
            </div>
          ) : results.length ? (
            <div className="grid gap-4">
              {results.map((result) => (
                <a
                  key={`${result.locale}:${result.href}`}
                  href={result.href}
                  className="rounded-[1.75rem] border border-slate-200 bg-white/90 p-6 shadow-panel transition hover:border-azure hover:-translate-y-0.5"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-xl font-semibold tracking-tight text-ink">{result.title}</p>
                      <p className="mt-3 leading-7 text-slate-600">{result.description}</p>
                    </div>
                    <span className="rounded-full bg-mist px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-azure">
                      {result.section ?? result.kind}
                    </span>
                  </div>
                  <span className="mt-5 inline-flex text-sm font-semibold text-azure">{copy.open}</span>
                </a>
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
