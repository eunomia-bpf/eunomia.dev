import type { BlogEntry } from "../lib/content/types";
import { formatDate as formatDisplayDate } from "../lib/dates";
import type { Locale } from "../lib/site-data";

type BlogListingProps = {
  title: string;
  description: string;
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

/** Groups blog entries by year, returning years in descending order. */
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

export function BlogListing({ title, description, entries, locale }: BlogListingProps) {
  const grouped = groupByYear(entries);

  return (
    <section className="pb-16">
      {/* Page header */}
      <div className="mb-10">
        <h1 className="text-3xl font-semibold tracking-tight text-ink md:text-[2.2rem]">{title}</h1>
        {description && (
          <p className="mt-4 max-w-2xl text-base leading-7 text-slate-600">{description}</p>
        )}
      </div>

      {/* Posts grouped by year */}
      <div className="space-y-12">
        {grouped.map(({ year, posts }) => (
          <div key={year}>
            <h2 className="mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
              {year}
            </h2>
            <div className="divide-y divide-slate-200 rounded-xl border border-slate-200 bg-white">
              {posts.map((post) => (
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
                      {post.description && (
                        <p className="mt-1 text-sm leading-5 text-slate-500 line-clamp-2">
                          {post.description}
                        </p>
                      )}
                    </div>
                    <time
                      dateTime={post.date}
                      className="mt-0.5 shrink-0 whitespace-nowrap text-xs text-slate-400"
                    >
                      {formatDate(post.date, locale)}
                    </time>
                  </div>
                </a>
              ))}
            </div>
          </div>
        ))}
      </div>

      {entries.length === 0 && (
        <p className="text-slate-500">No posts found.</p>
      )}
    </section>
  );
}
