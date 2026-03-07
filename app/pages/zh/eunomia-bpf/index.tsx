import { CardGrid } from "../../../components/CardGrid";
import { SeoHead } from "../../../components/SeoHead";
import { SiteChrome } from "../../../components/SiteChrome";
import { canonicalAlternates } from "../../../lib/seo";

const cards = [
  {
    title: "eunomia-bpf 入口",
    description: "保留项目中文路由和基础 SEO。",
    href: "/eunomia-bpf/",
    badge: "入口页"
  }
];

export default function EunomiaBpfZhPage() {
  return (
    <>
      <SeoHead
        title="eunomia-bpf 文档"
        description="保留 eunomia-bpf 中文入口页与多语言元数据。"
        path="/zh/eunomia-bpf/"
        alternates={canonicalAlternates("/eunomia-bpf/", "/zh/eunomia-bpf/")}
      />
      <SiteChrome
        locale="zh"
        eyebrow="eunomia-bpf"
        title="中文项目入口已经保留。"
        intro="这一层先解决 URL、SEO 和导航，再继续完整接入内容。"
      >
        <CardGrid cards={cards} />
      </SiteChrome>
    </>
  );
}
