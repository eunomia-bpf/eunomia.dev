import type { Locale } from "./site-data";

function toDate(value: string | undefined): Date | null {
  if (!value) {
    return null;
  }

  const normalized = /^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T00:00:00Z` : value;
  const date = new Date(normalized);
  return Number.isNaN(date.valueOf()) ? null : date;
}

export function formatDate(value: string | undefined, locale: Locale, options: Intl.DateTimeFormatOptions) {
  const date = toDate(value);
  if (!date) {
    return null;
  }

  return new Intl.DateTimeFormat(locale === "zh" ? "zh-CN" : "en-US", {
    timeZone: "UTC",
    ...options
  }).format(date);
}

export function toArticleDateTime(value: string | undefined) {
  return /^\d{4}-\d{2}-\d{2}$/.test(value ?? "") ? `${value}T00:00:00.000Z` : value;
}
