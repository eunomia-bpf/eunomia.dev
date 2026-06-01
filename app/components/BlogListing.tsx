import type { BlogEntry } from "../lib/content/types";
import type { Locale } from "../lib/site-data";
import { BlogPostList, BlogPostYearGroups } from "./BlogPostList";

type BlogListingProps = {
  title: string;
  description: string;
  entries: BlogEntry[];
  locale: Locale;
};

export function BlogListing({ title, description, entries, locale }: BlogListingProps) {
  const cleanDescription = description.trim();
  const showDescription = cleanDescription.length > 0 && cleanDescription !== title.trim();
  const featuredEntries = entries.slice(0, 3);
  const archiveEntries = entries.slice(3);
  const copy =
    locale === "zh"
      ? {
          latest: "最新文章",
          archive: "全部文章",
          empty: "没有找到文章。"
        }
      : {
          latest: "Latest writing",
          archive: "All posts",
          empty: "No posts found."
        };

  return (
    <section className="pb-16">
      <div className="border-b border-slate-200 pb-8">
        <h1 className="max-w-4xl text-4xl font-semibold tracking-normal text-ink md:text-5xl">{title}</h1>
        {showDescription && (
          <p className="mt-5 max-w-3xl text-base leading-7 text-slate-600">{cleanDescription}</p>
        )}
      </div>

      {featuredEntries.length ? (
        <section className="py-8" aria-labelledby="blog-featured">
          <h2 id="blog-featured" className="mb-4 text-2xl font-semibold tracking-normal text-ink">
            {copy.latest}
          </h2>
          <div className="grid gap-4 lg:grid-cols-3">
            {featuredEntries.map((entry) => (
              <BlogPostList key={entry.key} entries={[entry]} locale={locale} />
            ))}
          </div>
        </section>
      ) : null}

      {archiveEntries.length ? (
        <section className="border-t border-slate-200 pt-8" aria-labelledby="blog-archive">
          <h2 id="blog-archive" className="mb-5 text-2xl font-semibold tracking-normal text-ink">
            {copy.archive}
          </h2>
          <BlogPostYearGroups entries={entries} locale={locale} />
        </section>
      ) : null}

      {entries.length === 0 ? <p className="pt-8 text-slate-500">{copy.empty}</p> : null}
    </section>
  );
}
