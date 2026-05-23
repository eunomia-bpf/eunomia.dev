"use client";

import { useEffect, useRef, useState } from "react";

import { SearchBox } from "../components/SearchBox";
import { SidebarNav } from "../components/SidebarNav";
import type { SidebarGroup } from "../lib/content/types";
import { localizePath, normalizePath } from "../lib/paths";
import { getPrimaryNav } from "../lib/site-ia";
import type { Locale } from "../lib/site-data";
import { mobileNavCopyByLocale } from "../lib/ui-copy";

const ANIMATION_DURATION_MS = 300;
const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])'
].join(",");

type MobileNavProps = {
  locale: Locale;
  currentPath?: string;
  sidebar?: SidebarGroup[];
};

export function MobileNav({ locale, currentPath, sidebar }: MobileNavProps) {
  const [open, setOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const nav = getPrimaryNav(locale);
  const copy = mobileNavCopyByLocale[locale];
  const normalizedCurrentPath = normalizePath(currentPath);
  const childNavGroups = nav
    .filter((item) => item.children?.length)
    .map((item) => ({
      title: item.label,
      items: item.children?.map((child) => ({
        title: child.label,
        href: child.href
      })) ?? []
    }));
  const drawerGroups: SidebarGroup[] = [
    {
      title: copy.sections,
      items: nav.map((item) => ({
        title: item.label,
        href: item.href
      }))
    },
    ...childNavGroups,
    ...(sidebar?.filter((group) => group.items.length) ?? [])
  ];

  useEffect(() => {
    if (open) {
      if (closeTimerRef.current !== null) {
        clearTimeout(closeTimerRef.current);
        closeTimerRef.current = null;
      }
      setMounted(true);
    } else {
      closeTimerRef.current = setTimeout(() => {
        setMounted(false);
        closeTimerRef.current = null;
      }, ANIMATION_DURATION_MS);
    }
    return () => {
      if (closeTimerRef.current !== null) {
        clearTimeout(closeTimerRef.current);
      }
    };
  }, [open]);

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    const getFocusableElements = () =>
      [...(panelRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR) ?? [])].filter(
        (element) => element.getClientRects().length > 0
      );

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
        buttonRef.current?.focus();
        return;
      }

      if (event.key !== "Tab") {
        return;
      }

      const focusable = getFocusableElements();
      const first = focusable[0];
      const last = focusable.at(-1);
      if (!first || !last) {
        event.preventDefault();
        return;
      }

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    const focusFrame = window.requestAnimationFrame(() => {
      getFocusableElements()[0]?.focus();
    });

    window.addEventListener("keydown", handleKeyDown);
    const previousDocumentOverflow = document.documentElement.style.overflow;
    const previousBodyOverflow = document.body.style.overflow;
    document.documentElement.style.overflow = "hidden";
    document.body.style.overflow = "hidden";

    return () => {
      window.cancelAnimationFrame(focusFrame);
      window.removeEventListener("keydown", handleKeyDown);
      document.documentElement.style.overflow = previousDocumentOverflow;
      document.body.style.overflow = previousBodyOverflow;
    };
  }, [open]);

  return (
    <div className="relative lg:hidden">
      <button
        ref={buttonRef}
        type="button"
        aria-expanded={open}
        aria-controls="mobile-nav-panel"
        aria-label={open ? copy.close : copy.open}
        className="inline-flex h-11 w-11 items-center justify-center rounded-xl border border-slate-200 text-slate-700 transition hover:border-slate-300 hover:text-ink"
        onClick={() => setOpen((value) => !value)}
      >
        <span className="text-lg leading-none">{open ? "\u00d7" : "\u2261"}</span>
      </button>
      {mounted ? (
        <div id="mobile-nav-overlay" className="fixed inset-0 z-50 lg:hidden">
          <button
            type="button"
            aria-label={copy.close}
            className={[
              "absolute inset-0 bg-slate-900/45",
              "transition-opacity duration-300 ease-out",
              open ? "opacity-100" : "opacity-0"
            ].join(" ")}
            onClick={() => setOpen(false)}
          />
          <div
            ref={panelRef}
            id="mobile-nav-panel"
            className={[
              "absolute left-0 top-0 flex h-full w-[min(24rem,calc(100vw-1.5rem))] max-w-full flex-col border-r border-slate-200 bg-white shadow-2xl",
              "transition-transform duration-300 ease-out",
              open ? "translate-x-0" : "-translate-x-full"
            ].join(" ")}
          >
            <div className="flex items-center justify-between border-b border-slate-200 px-4 py-4">
              <a
                href={localizePath("/", locale)}
                className="text-base font-semibold tracking-tight text-ink"
                onClick={() => setOpen(false)}
              >
                eunomia.dev
              </a>
              <button
                type="button"
                aria-label={copy.close}
                className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-slate-200 text-slate-700 transition hover:border-slate-300 hover:text-ink"
                onClick={() => setOpen(false)}
              >
                <span className="text-lg leading-none">\u00d7</span>
              </button>
            </div>
            <div className="flex-1 overflow-y-auto px-4 py-4">
              <SearchBox
                locale={locale}
                containerClassName="block"
                inputClassName="w-full border-slate-300 bg-slate-50"
                panelClassName="left-0 right-0 w-auto"
                onNavigate={() => setOpen(false)}
              />
              <SidebarNav
                groups={drawerGroups}
                currentPath={normalizedCurrentPath}
                variant="mobile"
                navLabel={copy.navigation}
                className="mt-6 pb-6"
                onNavigate={() => setOpen(false)}
              />
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
