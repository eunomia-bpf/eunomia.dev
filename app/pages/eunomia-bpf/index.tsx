import { CardGrid } from "../../components/CardGrid";
import { SeoHead } from "../../components/SeoHead";
import { SiteChrome } from "../../components/SiteChrome";
import { canonicalAlternates } from "../../lib/seo";

const cards = [
  {
    title: "Packaging and distribution",
    description: "Keep the route and entry point intact during the frontend migration.",
    href: "/tutorials/",
    badge: "CO-RE"
  }
];

export default function EunomiaBpfPage() {
  return (
    <>
      <SeoHead
        title="eunomia-bpf"
        description="Stable entry page for eunomia-bpf in the custom frontend prototype."
        path="/eunomia-bpf/"
        alternates={canonicalAlternates("/eunomia-bpf/", "/zh/eunomia-bpf/")}
      />
      <SiteChrome
        locale="en"
        eyebrow="eunomia-bpf"
        title="Distribution-oriented eBPF tooling keeps its route."
        intro="This page is a placeholder shell for the eventual Markdown-driven project documentation."
      >
        <CardGrid cards={cards} />
      </SiteChrome>
    </>
  );
}
