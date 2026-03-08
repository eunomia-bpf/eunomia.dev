import { SearchResults } from "../components/SearchResults";
import type { LandingPageData, MarkdownPage, SearchResult, SidebarGroup } from "./content/types";
import { CollectionPageView, HomePageView, type CollectionPageProps, type HomePageData, SectionPageView } from "./page-factories";
import type { Locale } from "./site-data";

type SearchPageProps = {
  query: string;
  results: SearchResult[];
  sidebar: SidebarGroup[];
};

type SectionPageProps = {
  page: MarkdownPage;
  section: string;
};

export function createHomePage(locale: Locale, eyebrow: string) {
  return function HomePage({ page }: { page: HomePageData }) {
    return <HomePageView page={page} locale={locale} eyebrow={eyebrow} />;
  };
}

export function createCollectionPage<IndexPage extends LandingPageData, ArticlePage extends MarkdownPage>(
  locale: Locale,
  eyebrow: string
) {
  return function CollectionPage(props: CollectionPageProps<IndexPage, ArticlePage>) {
    return <CollectionPageView {...props} locale={locale} eyebrow={eyebrow} />;
  };
}

export function createSectionPage(locale: Locale) {
  return function SectionPage({ page, section }: SectionPageProps) {
    return <SectionPageView page={page} section={section} locale={locale} />;
  };
}

export function createSearchPage(locale: Locale) {
  return function SearchPage({ query, results, sidebar }: SearchPageProps) {
    return <SearchResults locale={locale} query={query} results={results} sidebar={sidebar} />;
  };
}

export function createFeedPage() {
  return function FeedPage() {
    return null;
  };
}
