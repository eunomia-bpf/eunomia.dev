"use client";

import { useState } from "react";

import { SearchBox } from "../components/SearchBox";
import { navByLocale, type Locale } from "../lib/site-data";

type MobileNavProps = {
  locale: Locale;
};

export function MobileNav({ locale }: MobileNavProps) {
  const [open, setOpen] = useState(false);
  const nav = navByLocale[locale];
  const copy =
    locale === "zh"
      ? {
          open: "打开导航",
          close: "关闭导航"
        }
      : {
          open: "Open navigation",
          close: "Close navigation"
        };

  return (
    <div className="relative md:hidden">
      <button
        type="button"
        aria-expanded={open}
        aria-controls="mobile-nav-panel"
        aria-label={open ? copy.close : copy.open}
        className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-slate-200 text-slate-700 transition hover:border-azure hover:text-azure"
        onClick={() => setOpen((value) => !value)}
      >
        <span className="text-lg leading-none">{open ? "\u00d7" : "\u2261"}</span>
      </button>
      {open ? (
        <div
          id="mobile-nav-panel"
          className="absolute right-0 top-[calc(100%+0.75rem)] z-50 w-[min(22rem,calc(100vw-2.5rem))] rounded-[1.75rem] border border-slate-200 bg-white/95 p-5 shadow-lg backdrop-blur"
        >
          <SearchBox
            locale={locale}
            containerClassName="block"
            inputClassName="w-full"
            panelClassName="left-0 right-0 w-auto"
            onNavigate={() => setOpen(false)}
          />
          <nav className="mt-4 grid gap-2">
            {nav.map((item) => (
              <a
                key={item.href}
                href={item.href}
                className="rounded-2xl border border-slate-200 px-4 py-3 text-sm font-medium text-slate-700 transition hover:border-azure hover:text-azure"
                onClick={() => setOpen(false)}
              >
                {item.label}
              </a>
            ))}
          </nav>
        </div>
      ) : null}
    </div>
  );
}
