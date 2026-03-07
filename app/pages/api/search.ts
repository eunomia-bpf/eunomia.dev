import type { NextApiRequest, NextApiResponse } from "next";

import { searchContent } from "../../lib/content/search";
import type { SearchResult } from "../../lib/content/types";
import type { Locale } from "../../lib/site-data";

type SearchResponse = {
  results: SearchResult[];
};

function normalizeLocale(value: string | string[] | undefined): Locale {
  return value === "zh" ? "zh" : "en";
}

function normalizeQuery(value: string | string[] | undefined): string {
  if (Array.isArray(value)) {
    return value[0] ?? "";
  }

  return value ?? "";
}

export default function handler(req: NextApiRequest, res: NextApiResponse<SearchResponse>) {
  const locale = normalizeLocale(req.query.locale);
  const query = normalizeQuery(req.query.q);
  const parsedLimit = Number.parseInt(normalizeQuery(req.query.limit), 10);
  const limit = Number.isFinite(parsedLimit) ? Math.max(1, Math.min(parsedLimit, 24)) : 8;

  res.setHeader("Cache-Control", "public, s-maxage=300, stale-while-revalidate=3600");
  res.status(200).json({
    results: searchContent(query, locale, limit)
  });
}
