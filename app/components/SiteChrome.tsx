import type { PropsWithChildren, ReactNode } from "react";

import type { LocaleAlternates, SidebarGroup } from "../lib/content/types";
import type { Locale } from "../lib/site-data";
import { DocsSidebar } from "./DocsSidebar";
import { SiteFooter } from "./SiteFooter";
import { SiteHeader } from "./SiteHeader";

type SiteChromeProps = PropsWithChildren<{
  locale: Locale;
  eyebrow: string;
  title: string;
  intro: string;
  hero?: ReactNode;
  leadMode?: "compact" | "none";
  currentPath?: string;
  sidebar?: SidebarGroup[];
  alternates?: LocaleAlternates;
}>;

export function SiteChrome({
  children,
  locale,
  eyebrow,
  title,
  intro,
  hero,
  leadMode = "compact",
  currentPath,
  sidebar,
  alternates
}: SiteChromeProps) {
  return (
    <div className="min-h-screen bg-white">
      <SiteHeader locale={locale} currentPath={currentPath} alternates={alternates} />
      <main className="pb-16">
        {hero ?? (
          leadMode === "compact" ? (
            <section className="border-b border-slate-200 bg-white">
              <div className="mx-auto max-w-[94rem] px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">{eyebrow}</p>
                <h1 className="mt-3 max-w-4xl text-3xl font-semibold tracking-tight text-ink md:text-4xl">
                  {title}
                </h1>
                <p className="mt-4 max-w-3xl text-base leading-7 text-slate-600 md:text-lg">{intro}</p>
              </div>
            </section>
          ) : null
        )}
        <div className="mx-auto max-w-[94rem] px-4 sm:px-6 lg:px-8">
          <div className={sidebar?.length ? "lg:grid lg:grid-cols-[17rem_minmax(0,1fr)] lg:gap-10" : ""}>
            {sidebar?.length ? (
              <DocsSidebar groups={sidebar} currentPath={currentPath ?? "/"} className="hidden lg:block lg:mt-8" />
            ) : null}
            <div className="min-w-0 pt-8">{children}</div>
          </div>
        </div>
      </main>
      <SiteFooter locale={locale} />
    </div>
  );
}
