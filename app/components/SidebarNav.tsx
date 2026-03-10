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

  return (
    <nav
      aria-label={navLabel}
      className={joinClassNames(isMobile ? "space-y-6" : "space-y-6 py-8", className)}
    >
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
                      "block border-l px-3 text-sm leading-6 transition",
                      isMobile ? "py-2" : "py-1",
                      isExact
                        ? "border-slate-900 font-medium text-ink"
                        : isAncestor
                          ? "border-slate-300 text-ink"
                          : "border-slate-200 text-slate-600 hover:border-slate-300 hover:text-ink"
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
