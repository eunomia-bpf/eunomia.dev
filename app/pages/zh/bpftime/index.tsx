import { CardGrid } from "../../../components/CardGrid";
import { SeoHead } from "../../../components/SeoHead";
import { SiteChrome } from "../../../components/SiteChrome";
import { canonicalAlternates } from "../../../lib/seo";

const cards = [
  {
    title: "bpftime 入口",
    description: "保留项目落地页、SEO 和多语言路径。",
    href: "/bpftime/",
    badge: "文档"
  }
];

export default function BpftimeZhPage() {
  return (
    <>
      <SeoHead
        title="bpftime 文档"
        description="保留 bpftime 中文入口页、固定路径与基础 SEO。"
        path="/zh/bpftime/"
        alternates={canonicalAlternates("/bpftime/", "/zh/bpftime/")}
      />
      <SiteChrome
        locale="zh"
        eyebrow="bpftime"
        title="中文项目页已经就位。"
        intro="这一页先保证路由、标题、canonical 和 hreflang 都稳定。"
      >
        <CardGrid cards={cards} />
      </SiteChrome>
    </>
  );
}
