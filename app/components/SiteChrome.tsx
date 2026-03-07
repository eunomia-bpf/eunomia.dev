import type { PropsWithChildren, ReactNode } from "react";

import type { Locale } from "../lib/site-data";
import { SiteFooter } from "./SiteFooter";
import { SiteHeader } from "./SiteHeader";

type SiteChromeProps = PropsWithChildren<{
  locale: Locale;
  eyebrow: string;
  title: string;
  intro: string;
  hero?: ReactNode;
}>;

export function SiteChrome({ children, locale, eyebrow, title, intro, hero }: SiteChromeProps) {
  return (
    <div className="min-h-screen">
      <SiteHeader locale={locale} />
      <main>
        {hero ?? (
          <section className="mx-auto max-w-6xl px-5 pb-8 pt-14">
            <p className="mb-4 text-sm font-semibold uppercase tracking-[0.24em] text-azure">{eyebrow}</p>
            <div className="rounded-[2rem] border border-white/60 bg-white/85 p-8 shadow-panel md:p-12">
              <h1 className="max-w-4xl text-4xl font-semibold tracking-tight text-ink md:text-6xl">
                {title}
              </h1>
              <p className="mt-5 max-w-3xl text-lg leading-8 text-slate-600">{intro}</p>
            </div>
          </section>
        )}
        {children}
      </main>
      <SiteFooter />
    </div>
  );
}
