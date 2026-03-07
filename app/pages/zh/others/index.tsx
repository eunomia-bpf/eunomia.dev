import { CardGrid } from "../../../components/CardGrid";
import { SeoHead } from "../../../components/SeoHead";
import { SiteChrome } from "../../../components/SiteChrome";
import { canonicalAlternates } from "../../../lib/seo";

const cards = [
  {
    title: "生态入口",
    description: "保留生态页面的中文路径和元数据。",
    href: "/others/",
    badge: "生态"
  }
];

export default function OthersZhPage() {
  return (
    <>
      <SeoHead
        title="生态"
        description="保留生态页面的中文入口、路径与 SEO。"
        path="/zh/others/"
        alternates={canonicalAlternates("/others/", "/zh/others/")}
      />
      <SiteChrome
        locale="zh"
        eyebrow="生态"
        title="生态页中文入口已经保留。"
        intro="后续可以在这一层继续接 talks、papers 和项目矩阵。"
      >
        <CardGrid cards={cards} />
      </SiteChrome>
    </>
  );
}
