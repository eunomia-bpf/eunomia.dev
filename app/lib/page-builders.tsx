import { useRouter } from "next/router";
import { useDeferredValue, useEffect, useState } from "react";

import { SearchResults } from "../components/SearchResults";
import type { DocsPage, SearchResult, SidebarGroup } from "./content/types";
import { loadSearchDocuments, searchDocuments } from "./client-search";
import { DocsPageView, HomePageView, type HomePageData } from "./page-factories";
import type { Locale } from "./site-data";

type SearchPageProps = {
  sidebar: SidebarGroup[];
};

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
