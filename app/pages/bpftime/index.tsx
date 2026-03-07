import { CardGrid } from "../../components/CardGrid";
import { SeoHead } from "../../components/SeoHead";
import { SiteChrome } from "../../components/SiteChrome";
import { canonicalAlternates } from "../../lib/seo";

const cards = [
  {
    title: "Userspace eBPF runtime",
    description: "Keep the `bpftime` route alive while the content pipeline is rebuilt.",
    href: "/blog/",
    badge: "Runtime"
  },
  {
    title: "Examples",
    description: "Point readers back to tutorials and implementation notes during migration.",
    href: "/tutorials/",
    badge: "Examples"
  }
];

export default function BpftimePage() {
  return (
    <>
      <SeoHead
        title="bpftime"
        description="Stable bpftime route with canonical and alternate metadata in the custom frontend app."
        path="/bpftime/"
        alternates={canonicalAlternates("/bpftime/", "/zh/bpftime/")}
      />
      <SiteChrome
        locale="en"
        eyebrow="bpftime"
        title="Userspace eBPF runtime pages can move first without breaking SEO."
        intro="This page keeps a stable landing route for bpftime while the custom frontend, docs ingestion, and richer information architecture are built out."
      >
        <CardGrid cards={cards} />
      </SiteChrome>
    </>
  );
}
