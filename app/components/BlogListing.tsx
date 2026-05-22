import type { BlogEntry } from "../lib/content/types";
import type { Locale } from "../lib/site-data";
import { BlogPostYearGroups } from "./BlogPostList";

type BlogListingProps = {
  title: string;
  description: string;
  entries: BlogEntry[];
  locale: Locale;
};

export function BlogListing({ title, description, entries, locale }: BlogListingProps) {
  return (
    <section className="pb-16">
      <div className="mb-10">
        <h1 className="text-3xl font-semibold tracking-tight text-ink md:text-[2.2rem]">{title}</h1>
        {description && (
          <p className="mt-4 max-w-2xl text-base leading-7 text-slate-600">{description}</p>
        )}
      </div>

      <BlogPostYearGroups entries={entries} locale={locale} />

      {entries.length === 0 && (
        <p className="text-slate-500">No posts found.</p>
      )}
    </section>
  );
}
