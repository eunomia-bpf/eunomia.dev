import type { Locale } from "../site-data";
import type { ContentManifestRecord } from "./types";

export function resolveRecordSource(record: ContentManifestRecord, locale: Locale): string | null {
  return record.sourceByLocale[locale] ?? record.sourceByLocale.en ?? record.sourceByLocale.zh ?? null;
}

export function resolveRecordHref(record: ContentManifestRecord, locale: Locale): string | null {
  return record.routeByLocale[locale] ?? record.routeByLocale.en ?? record.routeByLocale.zh ?? null;
}
