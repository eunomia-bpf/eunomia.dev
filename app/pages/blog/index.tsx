import { CardGrid } from "../../components/CardGrid";
import { SeoHead } from "../../components/SeoHead";
import { SiteChrome } from "../../components/SiteChrome";
import { blogPath, blogPosts } from "../../lib/site-data";
import { canonicalAlternates } from "../../lib/seo";

export default function BlogIndexPage() {
  const cards = blogPosts.map((post) => ({
    title: post.title,
    description: post.excerpt,
    href: blogPath(post),
    badge: `${post.year}-${post.month}-${post.day}`
  }));

  return (
    <>
      <SeoHead
        title="Blog"
        description="A custom frontend blog index that preserves the public URL structure used by eunomia.dev."
        path="/blog/"
        alternates={canonicalAlternates("/blog/", "/zh/blog/")}
      />
      <SiteChrome
        locale="en"
        eyebrow="Blog"
        title="Dated permalinks stay stable."
        intro="This page is the first compatibility slice for the blog system: static HTML, canonical metadata, and links that keep the current permalink shape."
      >
        <CardGrid cards={cards} />
      </SiteChrome>
    </>
  );
}
