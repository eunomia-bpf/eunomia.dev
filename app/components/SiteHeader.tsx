import { MobileNav } from "../components/MobileNav";
import { SearchBox } from "../components/SearchBox";
import type { LocaleAlternates } from "../lib/content/types";
import { localizePath } from "../lib/paths";
import { navByLocale, type Locale } from "../lib/site-data";

type SiteHeaderProps = {
  locale: Locale;
  currentPath?: string;
  alternates?: LocaleAlternates;
};

function normalizePath(pathname: string | undefined): string {
  if (!pathname) {
    return "/";
  }

  const normalized = pathname.endsWith("/") && pathname !== "/" ? pathname.slice(0, -1) : pathname;
  return normalized || "/";
}

function isActivePath(currentPath: string, href: string): boolean {
  const normalizedCurrentPath = normalizePath(currentPath);
  const normalizedHref = normalizePath(href);

  if (normalizedHref === "/") {
    return normalizedCurrentPath === "/";
  }

  return normalizedCurrentPath === normalizedHref || normalizedCurrentPath.startsWith(`${normalizedHref}/`);
}

export function SiteHeader({ locale, currentPath, alternates }: SiteHeaderProps) {
  const nav = navByLocale[locale];
  const alternateLocale: Locale = locale === "en" ? "zh" : "en";
  const languageToggle = alternates?.[alternateLocale] ?? null;
  const normalizedCurrentPath = normalizePath(currentPath);

  return (
    <header className="sticky top-0 z-40 border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-[94rem] items-center justify-between gap-4 px-4 py-3 sm:px-6 lg:px-8">
        <a href={localizePath("/", locale)} className="min-w-0 text-base font-semibold tracking-tight text-ink">
          eunomia.dev
        </a>
        <nav className="hidden items-center gap-2 text-sm font-medium text-slate-600 lg:flex">
          {nav.map((item) => (
            <a
              key={item.href}
              href={item.href}
              aria-current={isActivePath(normalizedCurrentPath, item.href) ? "page" : undefined}
              className={`rounded-lg px-3 py-2 transition ${
                isActivePath(normalizedCurrentPath, item.href)
                  ? "bg-slate-100 text-ink"
                  : "hover:bg-slate-50 hover:text-ink"
              }`}
            >
              {item.label}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-2 sm:gap-3">
          <SearchBox locale={locale} containerClassName="hidden md:block" inputClassName="w-56" />
          {languageToggle ? (
            <a
              href={languageToggle}
              className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-700 transition hover:border-slate-300 hover:text-ink"
            >
              {locale === "en" ? "中文" : "EN"}
            </a>
          ) : (
            <span
              aria-disabled="true"
              className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-400"
            >
              {locale === "en" ? "中文" : "EN"}
            </span>
          )}
          <MobileNav locale={locale} currentPath={normalizedCurrentPath} />
        </div>
      </div>
    </header>
  );
}
