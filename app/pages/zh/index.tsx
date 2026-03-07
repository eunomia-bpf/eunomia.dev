import { CardGrid } from "../../components/CardGrid";
import { SeoHead } from "../../components/SeoHead";
import { SiteChrome } from "../../components/SiteChrome";
import { canonicalAlternates } from "../../lib/seo";

const cards = [
  {
    title: "教程",
    description: "保留教程入口、语言路径和基础 SEO 语义。",
    href: "/zh/tutorials/",
    badge: "兼容层"
  },
  {
    title: "博客",
    description: "保留博客列表与发布日期路径的前端壳。",
    href: "/zh/blog/",
    badge: "日期 URL"
  },
  {
    title: "文档",
    description: "保留 `bpftime`、`eunomia-bpf` 与生态入口页。",
    href: "/zh/bpftime/",
    badge: "多语言"
  }
];

export default function HomeZhPage() {
  return (
    <>
      <SeoHead
        title="释放 eBPF 的潜力"
        description="面向 eunomia.dev 的 React + Tailwind 自定义前端原型，优先保留 SEO、路径与内容入口。"
        path="/zh/"
        alternates={canonicalAlternates("/", "/zh/")}
      />
      <SiteChrome
        locale="zh"
        eyebrow="自定义前端"
        title="先保留兼容性，再逐步替换 MkDocs 渲染层。"
        intro="这一版先覆盖首页、教程、博客和项目入口，并把 SEO、语言切换、编辑链接与固定路径稳住。"
      >
        <CardGrid cards={cards} />
      </SiteChrome>
    </>
  );
}
