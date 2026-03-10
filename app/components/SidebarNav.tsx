"use client";

import { useEffect, useRef } from "react";

import type { SidebarGroup } from "../lib/content/types";
import { normalizePath } from "../lib/paths";
import { joinClassNames } from "../lib/utils";

type SidebarNavProps = {
  groups: SidebarGroup[];
  currentPath: string;
  variant?: "desktop" | "mobile";
  navLabel?: string;
  className?: string;
  onNavigate?: () => void;
};

export function SidebarNav({
  groups,
  currentPath,
  variant = "desktop",
  navLabel = "Section navigation",
  className,
  onNavigate
}: SidebarNavProps) {
  const normalizedCurrentPath = normalizePath(currentPath);
  const isMobile = variant === "mobile";
  const navRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const activeEl = navRef.current?.querySelector<HTMLElement>('[aria-current="page"]');
    if (activeEl) {
      activeEl.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, []);

  return (
    <nav
      ref={navRef}
      aria-label={navLabel}
      className={joinClassNames(isMobile ? "space-y-5" : "space-y-5 py-8", className)}
    >
      {groups.map((group, index) => (
        <section
          key={group.title}
          className={joinClassNames(
            index !== 0
              ? "border-t border-slate-100 pt-5"
              : undefined
          )}
        >
          <p className="mb-3 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">
            {group.title}
          </p>
          <ul className="space-y-1.5">
            {group.items.map((item) => {
              const normalizedHref = normalizePath(item.href);
              const isExact = normalizedCurrentPath === normalizedHref;
              const isAncestor =
                !isExact &&
                normalizedHref !== "/" &&
                normalizedCurrentPath.startsWith(`${normalizedHref}/`);

              return (
                <li key={item.href}>
                  <a
                    href={item.href}
                    aria-current={isExact ? "page" : undefined}
                    className={joinClassNames(
                      "block rounded-md border-l px-3 text-sm leading-6 transition",
                      isMobile ? "py-2.5" : "py-1",
                      isExact
                        ? "border-slate-900 bg-slate-50 font-medium text-ink"
                        : isAncestor
                          ? "border-slate-300 text-ink hover:bg-slate-50"
                          : "border-slate-200 text-slate-600 hover:border-slate-300 hover:bg-slate-50 hover:text-ink"
                    )}
                    style={{ marginLeft: `${Math.max((item.depth ?? 0) - 1, 0) * 12}px` }}
                    onClick={onNavigate}
                  >
                    {item.title}
                  </a>
                </li>
              );
            })}
          </ul>
        </section>
      ))}
    </nav>
  );
}
