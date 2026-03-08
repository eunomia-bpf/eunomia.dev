import type { LocaleAlternates } from "./content/types";
import { siteConfig } from "./site-data";

export const STATIC_OG_IMAGE_PATH = "/og/default.svg";

export type AlternateLink = {
  hrefLang: string;
  href: string;
};

export function absoluteUrl(path: string): string {
  return new URL(path, siteConfig.siteUrl).toString();
}

export function ogImageUrl(_title: string, _eyebrow?: string): string {
  // Pure static export cannot support per-page OG rendering at request time.
  // All pages intentionally share one build-time SVG card emitted into public/.
  return absoluteUrl(STATIC_OG_IMAGE_PATH);
}

export function canonicalAlternates(alternates: LocaleAlternates): AlternateLink[] {
  return Object.entries(alternates)
    .filter((entry): entry is [string, string] => typeof entry[1] === "string" && Boolean(entry[1]))
    .map(([hrefLang, path]) => ({
      hrefLang,
      href: absoluteUrl(path)
    }));
}
