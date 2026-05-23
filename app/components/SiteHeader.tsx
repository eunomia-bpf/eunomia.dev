import Image from "next/image";

import { MobileNav } from "../components/MobileNav";
import { SearchBox } from "../components/SearchBox";
import type { LocaleAlternates, SidebarGroup } from "../lib/content/types";
import { localizePath, normalizePath } from "../lib/paths";
import { getPrimaryNav } from "../lib/site-ia";
import { siteConfig, type Locale } from "../lib/site-data";

const LOGO_SRC = "/_content-assets/docs/assets/icon.svg";

type SiteHeaderProps = {
  locale: Locale;
  currentPath?: string;
  sidebar?: SidebarGroup[];
  alternates?: LocaleAlternates;
};

function isActivePath(currentPath: string, href: string): boolean {
  const normalizedCurrentPath = normalizePath(currentPath);
  const normalizedHref = normalizePath(href);

  if (normalizedHref === "/") {
    return normalizedCurrentPath === "/";
  }

  return normalizedCurrentPath === normalizedHref || normalizedCurrentPath.startsWith(`${normalizedHref}/`);
}

export function SiteHeader({ locale, currentPath, sidebar, alternates }: SiteHeaderProps) {
  const nav = getPrimaryNav(locale);
  const alternateLocale: Locale = locale === "en" ? "zh" : "en";
  const languageToggle = alternates?.[alternateLocale] ?? null;
  const normalizedCurrentPath = normalizePath(currentPath);

  return (
    <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/95">
      <div className="mx-auto flex max-w-[82rem] items-center justify-between gap-4 px-4 py-3 sm:px-6 lg:px-8">
        <a
          href={localizePath("/", locale)}
          className="flex min-w-0 items-center gap-2 text-base font-semibold tracking-normal text-ink"
        >
          <Image
            src={LOGO_SRC}
            alt=""
            width={28}
            height={28}
            className="h-7 w-7 shrink-0 rounded-md"
            unoptimized
          />
          <span className="truncate">eunomia</span>
        </a>
        <nav className="hidden items-center gap-1 text-sm font-medium text-slate-600 lg:flex">
          {nav.map((item) => {
            const active = isActivePath(normalizedCurrentPath, item.href);
            const itemClassName = `inline-flex items-center gap-1.5 rounded-md px-3 py-2 transition ${
              active
                ? "bg-slate-100 text-ink"
                : "hover:bg-slate-50 hover:text-ink"
            }`;

            if (item.children?.length) {
              return (
                <div key={item.href} className="group relative">
                  <a
                    href={item.href}
                    aria-current={active ? "page" : undefined}
                    aria-haspopup="true"
                    className={itemClassName}
                  >
                    <span>{item.label}</span>
                    <span aria-hidden="true" className="text-xs text-slate-400 transition group-hover:text-slate-600">
                      ▾
                    </span>
                  </a>
                  <div className="invisible absolute left-0 top-full z-50 w-56 pt-2 opacity-0 transition group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100">
                    <div className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-lg">
                      {item.children.map((child) => (
                        <a
                          key={child.href}
                          href={child.href}
                          className="block border-t border-slate-100 px-4 py-3 text-sm text-slate-600 transition first:border-t-0 hover:bg-slate-50 hover:text-ink"
                        >
                          {child.label}
                        </a>
                      ))}
                    </div>
                  </div>
                </div>
              );
            }

            return (
              <a
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={itemClassName}
              >
                {item.label}
              </a>
            );
          })}
        </nav>
        <div className="flex items-center gap-2 sm:gap-3">
          <SearchBox locale={locale} containerClassName="hidden xl:block" inputClassName="w-56 border-slate-300 bg-slate-50" />
          <a
            href={siteConfig.repoUrl}
            className="hidden rounded-md border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:text-ink sm:inline-flex"
          >
            GitHub
          </a>
          {languageToggle ? (
            <a
              href={languageToggle}
              className="rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-700 transition hover:border-slate-400 hover:text-ink"
            >
              {locale === "en" ? "中文" : "EN"}
            </a>
          ) : (
            <span
              aria-disabled="true"
              className="rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-400"
            >
              {locale === "en" ? "中文" : "EN"}
            </span>
          )}
          <MobileNav locale={locale} currentPath={normalizedCurrentPath} sidebar={sidebar} />
        </div>
      </div>
    </header>
  );
}
