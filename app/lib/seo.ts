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

export function canonicalAlternates(enPath: string, zhPath: string): AlternateLink[] {
  return [
    { hrefLang: "en", href: absoluteUrl(enPath) },
    { hrefLang: "zh", href: absoluteUrl(zhPath) }
  ];
}
