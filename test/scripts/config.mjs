import fs from "node:fs";
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
  sectionIndex: "/bpftime/",
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

function parseScalar(value) {
  const trimmed = value.trim();
  if (
    (trimmed.startsWith("'") && trimmed.endsWith("'")) ||
    (trimmed.startsWith("\"") && trimmed.endsWith("\""))
  ) {
    return trimmed.slice(1, -1);
  }

  return trimmed;
}

function indentation(line) {
  return line.match(/^\s*/)?.[0].length ?? 0;
}

function readExpectedNavLabelsFromMkdocs() {
  const mkdocsPath = path.join(repoRoot, "mkdocs.yaml");
  const sections = [];
  let inSiteSections = false;
  let baseIndent = 0;
  let current = null;
  let nestedField = null;

  for (const line of fs.readFileSync(mkdocsPath, "utf8").split(/\r?\n/)) {
    if (!inSiteSections) {
      const siteSectionsMatch = line.match(/^(\s*)site_sections:\s*$/);
      if (siteSectionsMatch) {
        inSiteSections = true;
        baseIndent = siteSectionsMatch[1]?.length ?? 0;
      }
      continue;
    }

    if (line.trim() && indentation(line) <= baseIndent) {
      break;
    }

    if (!line.trim() || line.trim().startsWith("#")) {
      continue;
    }

    const sectionMatch = line.match(new RegExp(`^\\s{${baseIndent + 2}}(\\S[^:]*):\\s*$`));
    if (sectionMatch) {
      current = {
        label: null,
        nav: false,
        order: Number.POSITIVE_INFINITY
      };
      sections.push(current);
      nestedField = null;
      continue;
    }

    if (!current) {
      continue;
    }

    const fieldMatch = line.match(new RegExp(`^\\s{${baseIndent + 4}}([A-Za-z0-9_-]+):\\s*(.*)$`));
    if (fieldMatch) {
      const [, field, rawValue = ""] = fieldMatch;
      const value = parseScalar(rawValue);
      nestedField = null;

      if (field === "order") {
        current.order = Number(value);
        continue;
      }

      if (field === "labels" || field === "published") {
        nestedField = field;
      }
      continue;
    }

    const nestedMatch = line.match(new RegExp(`^\\s{${baseIndent + 6}}([A-Za-z0-9_-]+):\\s*(.*)$`));
    if (!nestedMatch || !nestedField) {
      continue;
    }

    const [, field, rawValue = ""] = nestedMatch;
    const value = parseScalar(rawValue);
    if (nestedField === "labels" && field === "en") {
      current.label = value;
    }
    if (nestedField === "published" && field === "nav") {
      current.nav = value.toLowerCase() === "true";
    }
  }

  const labels = sections
    .filter((section) => section.nav)
    .sort((left, right) => left.order - right.order)
    .map((section) => section.label)
    .filter(Boolean);

  if (!labels.length) {
    throw new Error(`Missing mkdocs nav labels in ${mkdocsPath}`);
  }

  return labels;
}

function readExpectedNavLabels() {
  const siteSectionsPath = path.join(appDir, ".generated", "content", "site-sections.json");
  if (!fs.existsSync(siteSectionsPath)) {
    return readExpectedNavLabelsFromMkdocs();
  }

  const payload = JSON.parse(fs.readFileSync(siteSectionsPath, "utf8"));
  const labels = payload.sections
    ?.filter((section) => section?.published?.nav)
    .sort((left, right) => left.order - right.order)
    .map((section) => section?.labels?.en)
    .filter(Boolean);

  if (!labels?.length) {
    throw new Error(`Missing generated nav labels in ${siteSectionsPath}`);
  }

  return labels;
}

export const expectedNavLabels = readExpectedNavLabels();
export const zhMarkers = ["教程", "主页", "文档", "博客"];
