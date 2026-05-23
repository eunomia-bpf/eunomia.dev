import type { BlogEntry } from "../lib/content/types";
import type { BlogLandingConfig } from "../lib/content/page-config";
import type { Locale } from "../lib/site-data";
import { BlogPostList, BlogPostYearGroups } from "./BlogPostList";

type BlogListingProps = {
  title: string;
  description: string;
  entries: BlogEntry[];
  landing?: BlogLandingConfig;
  locale: Locale;
};

export function BlogListing({ title, description, entries, landing, locale }: BlogListingProps) {
  const cleanDescription = description.trim();
  const showDescription = cleanDescription.length > 0 && cleanDescription !== title.trim();
  const featuredEntries = entries.slice(0, 3);
  const archiveEntries = entries.slice(3);
  const copy = landing?.sectionLabels;

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
            {copy?.featured[locale] ?? title}
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
            {copy?.archive[locale] ?? title}
          </h2>
          <BlogPostYearGroups entries={archiveEntries} locale={locale} />
        </section>
      ) : null}

      {entries.length === 0 ? <p className="pt-8 text-slate-500">{copy?.empty[locale] ?? ""}</p> : null}
    </section>
  );
}
