import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const testDir = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(testDir, "..");

export const baseUrl = new URL(process.env.BASE_URL ?? "https://eunomia.dev");
export const requestTimeoutMs = Number(process.env.REQUEST_TIMEOUT_MS ?? "15000");
export const maxPages = Number(process.env.MAX_PAGES ?? "50");
export const maxAssets = Number(process.env.MAX_ASSETS ?? "140");
export const appDir = process.env.APP_DIR ?? path.join(repoRoot, "app");
export const legacySitemapPath =
  process.env.LEGACY_SITEMAP_PATH ?? path.join(testDir, "fixtures", "legacy-sitemap.xml");
export const runtimeAuditDistDir = process.env.RUNTIME_AUDIT_DIST_DIR ?? ".next-runtime-audit";
export const rolloutAuditDistDir = process.env.ROLLOUT_AUDIT_DIST_DIR ?? ".next-rollout-audit";

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
  search: "/search/?q=hello%20world",
  tutorials: "/tutorials/",
  tutorialArticle: "/tutorials/1-helloworld/",
  tutorialNestedArticle: "/tutorials/38-btf-uprobe/test-verify/",
  blogIndex: "/blog/",
  blogArticle:
    "/blog/2026/02/17/agentcgroup-what-happens-when-ai-coding-agents-meet-os-resources/",
  legacyBlogArticle: "/blogs/bpftime/",
  sectionArticle: "/bpftime/llvmbpf/",
  mermaidArticle: "/eunomia-bpf/manual/",
  zhOnlySectionArticle: "/zh/eunomia-bpf/ecli/ecli-dockerfile-usage/",
  zhHome: "/zh/",
  zhSearch: "/zh/search/?q=hello%20world",
  zhTutorials: "/zh/tutorials/",
  zhLegacyBlogArticle: "/zh/blogs/bpftime/"
};

export const runtimeAuditRoutes = [
  "/tutorials/45-scx-nest/",
  "/zh/tutorials/45-scx-nest/",
  "/blog/2026/01/29/a-taxonomy-of-gpu-bugs-19-defect-classes-for-cuda-verification/",
  "/zh/blog/2026/01/29/a-taxonomy-of-gpu-bugs-19-defect-classes-for-cuda-verification/",
  "/others/nvbit-tutorial/nvbit-internals/",
  "/zh/others/nvbit-tutorial/nvbit-internals/"
];

export const rolloutAuditSampleBlogRoute =
  "/blog/2026/02/17/agentcgroup-what-happens-when-ai-coding-agents-meet-os-resources/";

export const expectedNavLabels = ["Tutorials", "Blog", "bpftime", "eBPF×AI/LLMs", "eunomia-bpf", "Ecosystem"];
export const zhMarkers = ["教程", "主页", "文档", "博客"];
