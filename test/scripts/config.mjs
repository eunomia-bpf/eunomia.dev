export const baseUrl = new URL(process.env.BASE_URL ?? "https://eunomia.dev");
export const requestTimeoutMs = Number(process.env.REQUEST_TIMEOUT_MS ?? "15000");
export const maxPages = Number(process.env.MAX_PAGES ?? "50");
export const maxAssets = Number(process.env.MAX_ASSETS ?? "140");
export const legacySitemapPath =
  process.env.LEGACY_SITEMAP_PATH ?? "/home/yunwei37/workspace/eunomia.dev/site/sitemap.xml";

export const crawlSeeds = [
  "/",
  "/zh/",
  "/tutorials/",
  "/tutorials/38-btf-uprobe/test-verify/",
  "/blog/",
  "/blogs/",
  "/zh/blogs/",
  "/bpftime/",
  "/bpftime/llvmbpf/",
  "/eunomia-bpf/",
  "/others/",
  "/others/miscellaneous/bpftime-roadmap/",
  "/GPTtrace/",
  "/wasm-bpf/"
];

export const smokeRoutes = {
  home: "/",
  tutorials: "/tutorials/",
  tutorialArticle: "/tutorials/1-helloworld/",
  tutorialNestedArticle: "/tutorials/38-btf-uprobe/test-verify/",
  blogIndex: "/blog/",
  blogArticle:
    "/blog/2026/02/17/agentcgroup-what-happens-when-ai-coding-agents-meet-os-resources/",
  legacyBlogArticle: "/blogs/bpftime/",
  sectionArticle: "/bpftime/llvmbpf/",
  zhHome: "/zh/",
  zhTutorials: "/zh/tutorials/",
  zhLegacyBlogArticle: "/zh/blogs/bpftime/"
};

export const expectedNavLabels = ["Tutorials", "Blog", "bpftime", "eunomia-bpf", "Ecosystem"];
export const zhMarkers = ["教程", "主页", "文档", "博客"];
