import { SearchResults } from "../components/SearchResults";
import type { LandingPageData, MarkdownPage, SearchResult, SidebarGroup } from "./content/types";
import { CollectionPageView, HomePageView, type CollectionPageProps, type HomePageData, SectionPageView } from "./page-factories";
import type { Locale } from "./site-data";

type SearchPageProps = {
  query: string;
  results: SearchResult[];
  sidebar: SidebarGroup[];
};

export type ContentPageProps =
  | {
      routeKind: "collection";
      eyebrow: string;
      content: CollectionPageProps<LandingPageData, MarkdownPage>;
    }
  | {
      routeKind: "section";
      section: string;
      page: MarkdownPage;
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
  return function ContentPage(props: ContentPageProps) {
    if (props.routeKind === "collection") {
      return <CollectionPageView {...props.content} locale={locale} eyebrow={props.eyebrow} />;
    }

    return <SectionPageView page={props.page} section={props.section} locale={locale} />;
  };
}

export function createFeedPage() {
  return function FeedPage() {
    return null;
  };
}
