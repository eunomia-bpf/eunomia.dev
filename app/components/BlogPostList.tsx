import type { BlogEntry } from "../lib/content/types";
import { formatDate as formatDisplayDate } from "../lib/dates";
import type { Locale } from "../lib/site-data";

type BlogPostListProps = {
  entries: BlogEntry[];
  locale: Locale;
};

function formatDate(dateStr: string | undefined, locale: Locale): string {
  return formatDisplayDate(dateStr, locale, {
    year: "numeric",
    month: "long",
    day: "numeric"
  }) ?? dateStr ?? "";
}

function buildBlogHref(entry: BlogEntry, locale: Locale): string {
  const prefix = locale === "zh" ? "/zh" : "";
  return `${prefix}/blog/${entry.year}/${entry.month}/${entry.day}/${entry.slug}/`;
}

function groupByYear(entries: BlogEntry[]): Array<{ year: string; posts: BlogEntry[] }> {
  const map = new Map<string, BlogEntry[]>();
  for (const entry of entries) {
    const year = entry.year ?? "Unknown";
    if (!map.has(year)) {
      map.set(year, []);
    }
    map.get(year)!.push(entry);
  }

  return [...map.entries()]
    .sort(([a], [b]) => b.localeCompare(a))
    .map(([year, posts]) => ({ year, posts }));
}

export function BlogPostList({ entries, locale }: BlogPostListProps) {
  return (
    <div className="divide-y divide-slate-200 rounded-lg border border-slate-200 bg-white">
      {entries.map((post) => {
        const description = post.description.trim();
        const showDescription = description.length > 0 && description !== post.title.trim();

        return (
          <a
            key={post.key}
            href={buildBlogHref(post, locale)}
            className="group block px-5 py-4 transition hover:bg-slate-50"
          >
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0 flex-1">
                <p className="font-medium tracking-tight text-ink group-hover:text-azure line-clamp-1 transition">
                  {post.title}
                </p>
                {showDescription && (
                  <p className="mt-1 text-sm leading-5 text-slate-500 line-clamp-2">
                    {description}
                  </p>
                )}
                {post.tags.length ? (
                  <ul className="mt-2 flex flex-wrap gap-1.5" aria-label={locale === "zh" ? "文章标签" : "Post tags"}>
                    {post.tags.map((tag) => (
                      <li key={tag} className="rounded-md border border-slate-200 px-1.5 py-0.5 text-[11px] font-medium text-slate-500">
                        {tag}
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>
              <time
                dateTime={post.date}
                className="mt-0.5 shrink-0 whitespace-nowrap text-xs text-slate-400"
              >
                {formatDate(post.date, locale)}
              </time>
            </div>
          </a>
        );
      })}
    </div>
  );
}

export function BlogPostYearGroups({ entries, locale }: BlogPostListProps) {
  return (
    <div className="space-y-12">
      {groupByYear(entries).map(({ year, posts }) => (
        <div key={year}>
          <h2 className="mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
            {year}
          </h2>
          <BlogPostList entries={posts} locale={locale} />
        </div>
      ))}
    </div>
  );
}
