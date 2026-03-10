import { useEffect, useState } from "react";

import type { HeadingEntry } from "../lib/content/types";
import { joinClassNames } from "../lib/utils";

type TableOfContentsProps = {
  headings: HeadingEntry[];
  compact?: boolean;
  className?: string;
  title?: string;
  ariaLabel?: string;
};

const indentByDepth: Record<number, string> = {
  2: "",
  3: "pl-4",
  4: "pl-8",
  5: "pl-10",
  6: "pl-12"
};

function useActiveHeading(headings: HeadingEntry[]): string {
  const [activeId, setActiveId] = useState<string>("");

  useEffect(() => {
    if (headings.length === 0) return;

    const headingIds = headings.map((h) => h.id);

    const observer = new IntersectionObserver(
      (entries) => {
        // Find the topmost visible heading
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);

        if (visible.length > 0) {
          setActiveId(visible[0].target.id);
        }
      },
      { rootMargin: "0px 0px -80% 0px", threshold: 0 }
    );

    const elements = headingIds
      .map((id) => document.getElementById(id))
      .filter((el): el is HTMLElement => el !== null);

    elements.forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, [headings]);

  return activeId;
}

export function TableOfContents({
  headings,
  compact = false,
  className,
  title = "On This Page",
  ariaLabel
}: TableOfContentsProps) {
  const activeId = useActiveHeading(headings);

  if (!headings.length) {
    return null;
  }

  return (
    <aside
      className={joinClassNames(
        compact
          ? "border border-slate-200 bg-slate-50/70 p-4"
          : "sticky top-24 self-start border-l border-slate-200 pl-5",
        className
      )}
    >
      <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">{title}</p>
      <nav aria-label={ariaLabel ?? title} className="mt-4">
        <ol className="space-y-2 text-sm text-slate-600">
          {headings.map((heading) => {
            const isActive = heading.id === activeId;
            return (
              <li key={heading.id} className={indentByDepth[heading.depth] ?? indentByDepth[6]}>
                <a
                  href={`#${heading.id}`}
                  className={joinClassNames(
                    "block py-0.5 transition hover:text-ink",
                    isActive
                      ? "font-semibold text-slate-900"
                      : "text-slate-500 hover:text-ink"
                  )}
                  aria-current={isActive ? "true" : undefined}
                >
                  {heading.text}
                </a>
              </li>
            );
          })}
        </ol>
      </nav>
    </aside>
  );
}
