export const baseUrl = new URL(process.env.BASE_URL ?? "https://eunomia.dev");
export const requestTimeoutMs = Number(process.env.REQUEST_TIMEOUT_MS ?? "15000");
export const maxPages = Number(process.env.MAX_PAGES ?? "50");
export const maxAssets = Number(process.env.MAX_ASSETS ?? "140");

export const seoPages = [
  { label: "home", path: "/", expectedLang: "en" },
  { label: "home zh", path: "/zh/", expectedLang: "zh" },
  { label: "tutorials", path: "/tutorials/", expectedLang: "en" },
  {
    label: "tutorial nested article",
    path: "/tutorials/38-btf-uprobe/test-verify/",
    expectedLang: "en"
  },
  { label: "blog", path: "/blog/", expectedLang: "en" },
  {
    label: "blog article",
    path: "/blog/2026/02/17/agentcgroup-what-happens-when-ai-coding-agents-meet-os-resources/",
    expectedLang: "en"
  },
  { label: "legacy blog", path: "/blogs/bpftime/", expectedLang: "en" },
  { label: "legacy blog zh", path: "/zh/blogs/bpftime/", expectedLang: "zh" },
  { label: "bpftime", path: "/bpftime/", expectedLang: "en" },
  { label: "bpftime llvmbpf", path: "/bpftime/llvmbpf/", expectedLang: "en" },
  { label: "others nested", path: "/others/miscellaneous/bpftime-roadmap/", expectedLang: "en" }
];

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
