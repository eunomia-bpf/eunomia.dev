import { SearchResults } from "../components/SearchResults";
import type { DocsPage, SearchResult, SidebarGroup } from "./content/types";
import { DocsPageView, HomePageView, type HomePageData } from "./page-factories";
import type { Locale } from "./site-data";

type SearchPageProps = {
  query: string;
  results: SearchResult[];
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
  return function SearchPage({ query, results, sidebar }: SearchPageProps) {
    return <SearchResults locale={locale} query={query} results={results} sidebar={sidebar} />;
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
