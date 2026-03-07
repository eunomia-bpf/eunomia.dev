"use client";

import { useEffect, useRef, useState } from "react";

import { SearchBox } from "../components/SearchBox";
import { navByLocale, type Locale } from "../lib/site-data";
import { mobileNavCopyByLocale } from "../lib/ui-copy";

type MobileNavProps = {
  locale: Locale;
};

export function MobileNav({ locale }: MobileNavProps) {
  const [open, setOpen] = useState(false);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const nav = navByLocale[locale];
  const copy = mobileNavCopyByLocale[locale];

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key !== "Escape") {
        return;
      }

      setOpen(false);
      buttonRef.current?.focus();
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  return (
    <div className="relative md:hidden">
      <button
        ref={buttonRef}
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
