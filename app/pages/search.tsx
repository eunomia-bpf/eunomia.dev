import type { GetServerSideProps } from "next";

import { SearchResults } from "../components/SearchResults";
import { searchContent } from "../lib/content/search";
import type { SearchResult } from "../lib/content/types";

type SearchPageProps = {
  query: string;
  results: SearchResult[];
};

function normalizeQuery(value: string | string[] | undefined): string {
  if (Array.isArray(value)) {
    return value[0] ?? "";
  }

  return value ?? "";
}

function serializeResults(results: SearchResult[]): SearchResult[] {
  return results.map((result) =>
    result.section
      ? result
      : {
          title: result.title,
          description: result.description,
          href: result.href,
          locale: result.locale,
          kind: result.kind
        }
  );
}

export const getServerSideProps: GetServerSideProps<SearchPageProps> = async ({ query }) => {
  const q = normalizeQuery(query.q);

  return {
    props: {
      query: q,
      results: serializeResults(searchContent(q, "en", 24))
    }
  };
};

export default function SearchPage({ query, results }: SearchPageProps) {
  return <SearchResults locale="en" query={query} results={results} />;
}
