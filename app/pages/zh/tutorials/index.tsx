import { CardGrid } from "../../../components/CardGrid";
import { SeoHead } from "../../../components/SeoHead";
import { SiteChrome } from "../../../components/SiteChrome";
import { canonicalAlternates } from "../../../lib/seo";

const cards = [
  {
    title: "教程总览",
    description: "保留教程入口和固定路径，作为后续 Markdown 渲染迁移的中文壳。",
    href: "/tutorials/1-helloworld/",
    badge: "教程"
  }
];

export default function TutorialsZhPage() {
  return (
    <>
      <SeoHead
        title="教程"
        description="保留教程入口、路径与基础 SEO 语义的中文页面。"
        path="/zh/tutorials/"
        alternates={canonicalAlternates("/tutorials/", "/zh/tutorials/")}
      />
      <SiteChrome
        locale="zh"
        eyebrow="教程"
        title="中文教程入口已经就位。"
        intro="这一页先验证语言路由、元数据和内部链接，下一步再把真实 Markdown 内容完整接过来。"
      >
        <CardGrid cards={cards} />
      </SiteChrome>
    </>
  );
}
