import type { HeadingEntry } from "../lib/content/types";

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

function joinClassNames(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}

export function TableOfContents({
  headings,
  compact = false,
  className,
  title = "On This Page",
  ariaLabel
}: TableOfContentsProps) {
  if (!headings.length) {
    return null;
  }

  return (
    <aside
      className={joinClassNames(
        compact
          ? "rounded-[1.5rem] border border-slate-200 bg-slate-50/90 p-5"
          : "sticky top-24 self-start rounded-[1.5rem] border border-white/70 bg-white/90 p-5 shadow-panel",
        className
      )}
    >
      <p className="text-xs font-semibold uppercase tracking-[0.24em] text-azure">{title}</p>
      <nav aria-label={ariaLabel ?? title} className="mt-4">
        <ol className="space-y-2 text-sm text-slate-600">
          {headings.map((heading) => (
            <li key={heading.id} className={indentByDepth[heading.depth] ?? indentByDepth[6]}>
              <a
                href={`#${heading.id}`}
                className="block rounded-lg px-2 py-1.5 transition hover:bg-slate-50 hover:text-azure"
              >
                {heading.text}
              </a>
            </li>
          ))}
        </ol>
      </nav>
    </aside>
  );
}
