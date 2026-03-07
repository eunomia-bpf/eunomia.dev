import { CardGrid } from "../../../components/CardGrid";
import { SeoHead } from "../../../components/SeoHead";
import { SiteChrome } from "../../../components/SiteChrome";
import { canonicalAlternates } from "../../../lib/seo";

const cards = [
  {
    title: "博客入口",
    description: "保留博客入口与日期路径规则，后续再接完整内容。",
    href: "/blog/2026/02/17/agentcgroup-what-happens-when-ai-coding-agents-meet-os-resources/",
    badge: "Blog"
  }
];

export default function BlogZhIndexPage() {
  return (
    <>
      <SeoHead
        title="博客"
        description="保留博客入口、多语言元数据和固定路径的中文页面。"
        path="/zh/blog/"
        alternates={canonicalAlternates("/blog/", "/zh/blog/")}
      />
      <SiteChrome
        locale="zh"
        eyebrow="博客"
        title="博客兼容层已经落地。"
        intro="这一步先把路由、SEO 和入口做稳，再继续补内容与搜索。"
      >
        <CardGrid cards={cards} />
      </SiteChrome>
    </>
  );
}
