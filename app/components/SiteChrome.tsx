import Head from "next/head";
import Script from "next/script";
import type { PropsWithChildren } from "react";

import { navByLocale, siteConfig, type Locale } from "../lib/site-data";

type SiteChromeProps = PropsWithChildren<{
  locale: Locale;
  eyebrow: string;
  title: string;
  intro: string;
}>;

export function SiteChrome({ children, locale, eyebrow, title, intro }: SiteChromeProps) {
  const nav = navByLocale[locale];
  const languageToggle = locale === "en" ? "/zh/" : "/";

  return (
    <>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <Script
        src={`https://www.googletagmanager.com/gtag/js?id=${siteConfig.analyticsId}`}
        strategy="beforeInteractive"
      />
      <Script
        id="ga-init"
        strategy="beforeInteractive"
        dangerouslySetInnerHTML={{
          __html: `window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', '${siteConfig.analyticsId}');`
        }}
      />
      <div className="min-h-screen">
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
              <label className="hidden md:block">
                <span className="sr-only">Search</span>
                <input
                  aria-label="Search"
                  type="text"
                  placeholder="Search"
                  className="w-40 rounded-full border border-slate-200 bg-slate-50 px-4 py-2 text-sm outline-none transition focus:border-azure"
                />
              </label>
              <a
                href={languageToggle}
                className="rounded-full border border-slate-200 px-3 py-2 text-sm text-slate-700 transition hover:border-azure hover:text-azure"
              >
                {locale === "en" ? "中文" : "EN"}
              </a>
            </div>
          </div>
        </header>
        <main>
          <section className="mx-auto max-w-6xl px-5 pb-8 pt-14">
            <p className="mb-4 text-sm font-semibold uppercase tracking-[0.24em] text-azure">{eyebrow}</p>
            <div className="rounded-[2rem] border border-white/60 bg-white/85 p-8 shadow-panel md:p-12">
              <h1 className="max-w-4xl text-4xl font-semibold tracking-tight text-ink md:text-6xl">
                {title}
              </h1>
              <p className="mt-5 max-w-3xl text-lg leading-8 text-slate-600">{intro}</p>
            </div>
          </section>
          {children}
        </main>
        <footer className="mt-16 border-t border-slate-200 bg-white/80">
          <div className="mx-auto flex max-w-6xl flex-col gap-4 px-5 py-10 text-sm text-slate-600 md:flex-row md:items-center md:justify-between">
            <p>Custom frontend migration slice for eunomia.dev.</p>
            <div className="flex gap-4">
              <a href={siteConfig.repoUrl} className="transition hover:text-azure">
                GitHub
              </a>
              <a href="https://github.com/orgs/eunomia-bpf/discussions" className="transition hover:text-azure">
                Discussion
              </a>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
}
