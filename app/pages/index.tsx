import { CardGrid } from "../components/CardGrid";
import { SeoHead } from "../components/SeoHead";
import { SiteChrome } from "../components/SiteChrome";
import { canonicalAlternates } from "../lib/seo";
import { homeSections } from "../lib/site-data";

export default function HomePage() {
  return (
    <>
      <SeoHead
        title="Unlock the potential of eBPF"
        description="A custom frontend prototype for eunomia.dev focused on preserving stable routes, SEO, and content entry points."
        path="/"
        alternates={canonicalAlternates("/", "/zh/")}
      />
      <SiteChrome
        locale="en"
        eyebrow="Custom Frontend"
        title="A React + Tailwind migration slice that keeps eunomia.dev crawlable."
        intro="This first implementation keeps the public route structure, metadata, and primary content entry points intact while moving toward a custom frontend stack."
      >
        <CardGrid cards={homeSections} />
      </SiteChrome>
    </>
  );
}
