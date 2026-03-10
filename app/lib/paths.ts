import type { Locale } from "./site-data";

export function normalizePath(pathname: string | undefined): string {
  if (!pathname || pathname === "/") {
    return "/";
  }

  return pathname.endsWith("/") ? pathname.slice(0, -1) : pathname;
}

export function localizePath(pathname: string, locale: Locale): string {
  const normalized =
    pathname.startsWith("/zh/") || pathname === "/zh" || pathname === "/zh/"
      ? pathname.replace(/^\/zh/, "") || "/"
      : pathname || "/";

  if (locale === "en") {
    return normalized;
  }

  return normalized === "/" ? "/zh/" : `/zh${normalized}`;
}
