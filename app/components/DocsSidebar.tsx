import type { SidebarGroup } from "../lib/content/types";
import { SidebarNav } from "./SidebarNav";

function joinClassNames(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}

type DocsSidebarProps = {
  groups: SidebarGroup[];
  currentPath: string;
  className?: string;
};

export function DocsSidebar({ groups, currentPath, className }: DocsSidebarProps) {
  return (
    <aside
      className={joinClassNames(
        "sticky top-20 hidden max-h-[calc(100vh-6rem)] overflow-y-auto pr-8 lg:block",
        className
      )}
    >
      <SidebarNav groups={groups} currentPath={currentPath} />
    </aside>
  );
}
