export const baseUrl = new URL(process.env.BASE_URL ?? "https://eunomia.dev");
export const requestTimeoutMs = Number(process.env.REQUEST_TIMEOUT_MS ?? "15000");
export const maxPages = Number(process.env.MAX_PAGES ?? "25");
export const maxAssets = Number(process.env.MAX_ASSETS ?? "80");

export const seoPages = [
  { label: "home", path: "/", expectedLang: "en" },
  { label: "home zh", path: "/zh/", expectedLang: "zh" },
  { label: "tutorials", path: "/tutorials/", expectedLang: "en" },
  { label: "blog", path: "/blog/", expectedLang: "en" },
  { label: "bpftime", path: "/bpftime/", expectedLang: "en" }
];

export const crawlSeeds = [
  "/",
  "/zh/",
  "/tutorials/",
  "/blog/",
  "/bpftime/",
  "/eunomia-bpf/",
  "/others/"
];

export const smokeRoutes = {
  home: "/",
  tutorials: "/tutorials/",
  tutorialArticle: "/tutorials/1-helloworld/",
  blogIndex: "/blog/",
  blogArticle:
    "/blog/2026/02/17/agentcgroup-what-happens-when-ai-coding-agents-meet-os-resources/",
  zhHome: "/zh/",
  zhTutorials: "/zh/tutorials/"
};

export const expectedNavLabels = ["Tutorials", "Blog", "bpftime", "eunomia-bpf"];
export const zhMarkers = ["教程", "主页", "文档", "博客"];
