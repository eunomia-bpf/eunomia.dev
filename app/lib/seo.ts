import type { LocaleAlternates } from "./content/types";
import { siteConfig } from "./site-data";

export type AlternateLink = {
  hrefLang: string;
  href: string;
};

export function absoluteUrl(path: string): string {
  return new URL(path, siteConfig.siteUrl).toString();
}

export function ogImageUrl(title: string, eyebrow?: string): string {
  const params = new URLSearchParams({
    title
  });

  if (eyebrow) {
    params.set("eyebrow", eyebrow);
  }

  return absoluteUrl(`/api/og?${params.toString()}`);
}

export function canonicalAlternates(alternates: LocaleAlternates): AlternateLink[] {
  return Object.entries(alternates)
    .filter((entry): entry is [string, string] => typeof entry[1] === "string" && Boolean(entry[1]))
    .map(([hrefLang, path]) => ({
      hrefLang,
      href: absoluteUrl(path)
    }));
}
