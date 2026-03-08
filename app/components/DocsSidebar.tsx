import type { SidebarGroup } from "../lib/content/types";

function normalizePath(pathname: string): string {
  if (!pathname || pathname === "/") {
    return "/";
  }

  return pathname.endsWith("/") ? pathname.slice(0, -1) : pathname;
}

function joinClassNames(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}

type DocsSidebarProps = {
  groups: SidebarGroup[];
  currentPath: string;
  className?: string;
};

export function DocsSidebar({ groups, currentPath, className }: DocsSidebarProps) {
  const normalizedCurrentPath = normalizePath(currentPath);

  return (
    <aside
      className={joinClassNames(
        "sticky top-20 hidden max-h-[calc(100vh-6rem)] overflow-y-auto pr-8 lg:block",
        className
      )}
    >
      <nav aria-label="Section navigation" className="space-y-6 py-8">
        {groups.map((group) => (
          <section key={group.title}>
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
                        "block border-l px-3 py-1 text-sm leading-6 transition",
                        isExact
                          ? "border-slate-900 font-medium text-ink"
                          : isAncestor
                            ? "border-slate-300 text-ink"
                            : "border-slate-200 text-slate-600 hover:border-slate-300 hover:text-ink"
                      )}
                      style={{ marginLeft: `${Math.max((item.depth ?? 0) - 1, 0) * 12}px` }}
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
    </aside>
  );
}
