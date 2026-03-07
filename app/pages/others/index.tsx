import { CardGrid } from "../../components/CardGrid";
import { SeoHead } from "../../components/SeoHead";
import { SiteChrome } from "../../components/SiteChrome";
import { canonicalAlternates } from "../../lib/seo";

const cards = [
  {
    title: "Talks and papers",
    description: "Keep an ecosystem landing page available while the full taxonomy is rebuilt.",
    href: "/blog/",
    badge: "Ecosystem"
  }
];

export default function OthersPage() {
  return (
    <>
      <SeoHead
        title="Ecosystem"
        description="Stable ecosystem route for talks, papers, and related project entry points."
        path="/others/"
        alternates={canonicalAlternates("/others/", "/zh/others/")}
      />
      <SiteChrome
        locale="en"
        eyebrow="Ecosystem"
        title="Ecosystem pages can move without losing route stability."
        intro="This placeholder route keeps the site structure coherent while content ingestion catches up."
      >
        <CardGrid cards={cards} />
      </SiteChrome>
    </>
  );
}
