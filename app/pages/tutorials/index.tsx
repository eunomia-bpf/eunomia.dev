import { CardGrid } from "../../components/CardGrid";
import { SeoHead } from "../../components/SeoHead";
import { SiteChrome } from "../../components/SiteChrome";
import { canonicalAlternates } from "../../lib/seo";
import { tutorialArticles, tutorialPath } from "../../lib/site-data";

export default function TutorialsPage() {
  const cards = tutorialArticles.map((article) => ({
    title: article.title,
    description: article.summary,
    href: tutorialPath(article.slug),
    badge: "Tutorial"
  }));

  return (
    <>
      <SeoHead
        title="Tutorials"
        description="Practical eBPF tutorials preserved behind a custom frontend shell."
        path="/tutorials/"
        alternates={canonicalAlternates("/tutorials/", "/zh/tutorials/")}
      />
      <SiteChrome
        locale="en"
        eyebrow="Tutorials"
        title="Learn eBPF through runnable examples."
        intro="This route intentionally stays simple: short summaries, stable links, and crawlable HTML while the full Markdown rendering pipeline is built out."
      >
        <CardGrid cards={cards} />
      </SiteChrome>
    </>
  );
}
