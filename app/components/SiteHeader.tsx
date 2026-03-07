import { SearchBox } from "../components/SearchBox";
import { navByLocale, type Locale } from "../lib/site-data";

type SiteHeaderProps = {
  locale: Locale;
};

export function SiteHeader({ locale }: SiteHeaderProps) {
  const nav = navByLocale[locale];
  const languageToggle = locale === "en" ? "/zh/" : "/";

  return (
    <header className="sticky top-0 z-40 border-b border-white/50 bg-white/85 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-6 px-5 py-4">
        <a href={locale === "en" ? "/" : "/zh/"} className="text-lg font-semibold tracking-tight">
          eunomia
        </a>
        <nav className="hidden items-center gap-5 text-sm font-medium text-slate-700 md:flex">
          {nav.map((item) => (
            <a key={item.href} href={item.href} className="transition hover:text-azure">
              {item.label}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-3">
          <SearchBox locale={locale} />
          <a
            href={languageToggle}
            className="rounded-full border border-slate-200 px-3 py-2 text-sm text-slate-700 transition hover:border-azure hover:text-azure"
          >
            {locale === "en" ? "中文" : "EN"}
          </a>
        </div>
      </div>
    </header>
  );
}
