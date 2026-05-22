import type { Locale } from "./site-data";
import { searchDocuments, type SearchDocument } from "./search-core";

type SearchResponse = {
  documents: SearchDocument[];
};

const searchIndexCache = new Map<Locale, Promise<SearchDocument[]>>();

export { searchDocuments };

export function loadSearchDocuments(locale: Locale): Promise<SearchDocument[]> {
  const cached = searchIndexCache.get(locale);
  if (cached) {
    return cached;
  }

  const pending = fetch(`/search-index/${locale}.json`).then(async (response) => {
    if (!response.ok) {
      throw new Error(`Failed to load static search index for ${locale}: ${response.status}`);
    }

    const payload = (await response.json()) as SearchResponse;
    if (!Array.isArray(payload.documents)) {
      throw new Error(`Invalid static search payload for ${locale}`);
    }

    return payload.documents;
  });

  searchIndexCache.set(locale, pending);
  return pending;
}
