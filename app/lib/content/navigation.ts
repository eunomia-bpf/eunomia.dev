import type { Locale } from "../site-data";
import { getContentManifest } from "./manifest";
import { parseMarkdown } from "./markdown";
import type { ContentManifestKind, ContentManifestRecord, PageContinuation, PageLink } from "./types";

function resolveRecordSource(record: ContentManifestRecord, locale: Locale): string | null {
  return record.sourceByLocale[locale] ?? record.sourceByLocale.en ?? record.sourceByLocale.zh ?? null;
}

function resolveRecordHref(record: ContentManifestRecord, locale: Locale): string | null {
  return record.routeByLocale[locale] ?? record.routeByLocale.en ?? record.routeByLocale.zh ?? null;
}

function recordToPageLink(record: ContentManifestRecord, locale: Locale): PageLink | null {
  const sourceRelative = resolveRecordSource(record, locale);
  const href = resolveRecordHref(record, locale);
  if (!sourceRelative || !href) {
    return null;
  }

  const parsed = parseMarkdown(sourceRelative);
  return {
    title: parsed.title,
    description: parsed.description,
    href
  };
}

export function buildIndexLink(sourceRelative: string | null, href: string | null): PageLink | undefined {
  if (!sourceRelative || !href) {
    return undefined;
  }

  const parsed = parseMarkdown(sourceRelative);
  return {
    title: parsed.title,
    description: parsed.description,
    href
  };
}

export function buildCollectionContinuation({
  kind,
  locale,
  currentPath,
  section,
  index
}: {
  kind: ContentManifestKind;
  locale: Locale;
  currentPath: string;
  section?: string;
  index?: PageLink;
}): PageContinuation | undefined {
  const links = getContentManifest()
    .filter((record) => record.kind === kind)
    .filter((record) => (section ? record.section === section : true))
    .filter((record) => Boolean(resolveRecordHref(record, locale)))
    .map((record) => recordToPageLink(record, locale))
    .filter((record): record is PageLink => Boolean(record));

  const currentIndex = links.findIndex((record) => record.href === currentPath);
  const continuation: PageContinuation = {};

  if (index && index.href !== currentPath) {
    continuation.index = index;
  }
  if (currentIndex > 0) {
    continuation.previous = links[currentIndex - 1];
  }
  if (currentIndex >= 0 && currentIndex < links.length - 1) {
    continuation.next = links[currentIndex + 1];
  }

  if (!continuation.index && !continuation.previous && !continuation.next) {
    return undefined;
  }

  return continuation;
}
