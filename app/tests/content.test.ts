import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { getStaticAssetPublicPath, writeStaticAssets } from "../lib/content/assets";
import { getBlogEntries, getBlogEntriesForLocale, getGenericSectionRouteEntries } from "../lib/content/collections";
import { loadHomePage } from "../lib/content/home-loader";
import { getContentModel } from "../lib/content/model";
import { renderFeed } from "../lib/content/feed";
import { getGitMetadata } from "../lib/content/git";
import {
  listRenderableRoutesForLocale,
  resolveAlternatesFromDocSource,
  resolveManifestRecordFromRoute
} from "../lib/content/manifest";
import { splitMaterialBlocks } from "../lib/content/material-blocks";
import {
  loadBlogIndex,
  loadBlogPage,
  loadLegacyBlogPage,
  loadSectionPage,
  loadTutorialPage,
  resolveContentPage
} from "../lib/content/loaders";
import { assertSupportedMarkdown, parseMarkdown } from "../lib/content/markdown";
import { getContentManifest } from "../lib/content/manifest";
import {
  readMkdocsHomeConfig,
  readMkdocsPrimaryNavChildren,
  readMkdocsSectionLandingPages,
  readMkdocsSectionPages,
  readMkdocsSectionSidebars,
  readMkdocsSiteMetadata,
  readMkdocsSiteSections,
  readMkdocsTopLevelNavSections
} from "../lib/content/mkdocs-config";
import { resolveCollectionPageSource } from "../lib/content/registry";
import { renderMarkdown, renderMarkdownBody, renderMarkdownDocument } from "../lib/content/render";
import { docPathToRoute, getGenericSectionRoutes, listSitemapRoutes } from "../lib/content/routes";
import { loadSearchDocuments, searchContent, writeSearchIndexes } from "../lib/content/search";
import { rewriteContentUrl } from "../lib/content/rewrite";
import { appRoot, siteRoot } from "../lib/content/roots";
import { resolveLocalizedSource, slugifyTitle } from "../lib/content/source";
import { absoluteUrl, ogImageUrl, STATIC_OG_IMAGE_PATH } from "../lib/seo";
import { siteConfig } from "../lib/site-data";
import { getPrimaryNav, getSectionSidebarOverride, getSiteSections } from "../lib/site-ia";
import { createContentPageRoute } from "../lib/route-builders";
import { buildSearchSidebar } from "../lib/content/sidebar";
import { writeStaticMetadata } from "../scripts/generate-static-metadata";

test("resolveLocalizedSource prefers zh variant when present", () => {
  assert.equal(
    resolveLocalizedSource("tutorials/38-btf-uprobe/README.md", "zh"),
    "tutorials/38-btf-uprobe/README.zh.md"
  );
});

test("resolveLocalizedSource falls back to english variant when zh variant is missing", () => {
  assert.equal(
    resolveLocalizedSource("eunomia-bpf/setup/README.md", "zh"),
    "eunomia-bpf/setup/README.en.md"
  );
});

test("docPathToRoute maps nested tutorial docs to public tutorial routes", () => {
  assert.equal(
    docPathToRoute("tutorials/38-btf-uprobe/test-verify/README.md", "en"),
    "/tutorials/38-btf-uprobe/test-verify/"
  );
});

test("docPathToRoute normalizes .en.md docs into stable public routes", () => {
  assert.equal(
    docPathToRoute("eunomia-bpf/setup/build.en.md", "en"),
    "/eunomia-bpf/setup/build/"
  );
});

test("slugifyTitle normalizes punctuation and accents into stable slugs", () => {
  assert.equal(
    slugifyTitle("AgentCgroup: What Happens When AI Coding Agents Meet OS Resources?"),
    "agentcgroup-what-happens-when-ai-coding-agents-meet-os-resources"
  );
  assert.equal(slugifyTitle("Caf\u00e9 Noir / 2026"), "cafe-noir-2026");
});

test("blog entries derive dated slugs from parsed metadata", () => {
  const entry = getBlogEntries().find((candidate) => candidate.key === "agentcgroup-characterization");
  assert.ok(entry);
  assert.equal(entry.slug, slugifyTitle(entry.title));
  assert.deepEqual([entry.year, entry.month, entry.day], ["2026", "02", "17"]);
  assert.equal(entry.sourceByLocale.en, "blog/posts/agentcgroup-characterization.md");
  assert.equal(entry.sourceByLocale.zh, undefined);
});

test("blog entries can pin historical slugs from front matter", () => {
  const entry = getBlogEntries().find((candidate) => candidate.key === "claude-code-analysis");

  assert.ok(entry);
  assert.equal(entry.slug, "reverse-engineering-claude-codes-ssl-traffic-with-ebpf");
});

test("home page data keeps markdown metadata but leaves layout to React", async () => {
  const home = await loadHomePage("en");
  const homeZh = await loadHomePage("zh");
  const expectedRecentPosts = getBlogEntriesForLocale("en").slice(0, 3);
  const expectedRecentPostsZh = getBlogEntriesForLocale("zh").slice(0, 3);
  const homeDescription =
    "Open-source eBPF systems research, userspace runtime tooling, AI-assisted tracing, and runnable Linux observability documentation.";

  assert.equal(home.description, homeDescription);
  assert.notEqual(home.description, home.title);
  assert.equal(Object.hasOwn(home, "bodyHtml"), false);
  assert.ok(!("cards" in home));
  assert.ok(!("moreLinks" in home));
  assert.equal(home.sourcePath, "https://github.com/eunomia-bpf/eunomia.dev/tree/main/docs/index.md");
  assert.deepEqual(home.recentPosts.map((post) => post.key), expectedRecentPosts.map((post) => post.key));
  assert.deepEqual(home.recentPosts.map((post) => post.description), expectedRecentPosts.map((post) => post.description));
  assert.ok(home.recentPosts.every((post) => post.description !== post.title));
  assert.equal(homeZh.sourcePath, "https://github.com/eunomia-bpf/eunomia.dev/tree/main/docs/index.zh.md");
  assert.equal(Object.hasOwn(homeZh, "bodyHtml"), false);
  assert.deepEqual(homeZh.recentPosts.map((post) => post.key), expectedRecentPostsZh.map((post) => post.key));
  assert.deepEqual(
    homeZh.recentPosts.map((post) => post.description),
    expectedRecentPostsZh.map((post) => post.description)
  );
  assert.ok(homeZh.recentPosts.every((post) => post.description !== post.title));
  assert.equal(home.home.projectsTitle.en, "Projects");
  assert.equal(home.home.projects[0]?.key, "bpftime");
  assert.equal(homeZh.home.projectsTitle.zh, "项目");
});

test("site metadata is sourced from mkdocs config", () => {
  const mkdocsMetadata = readMkdocsSiteMetadata();

  assert.equal(siteConfig.name, mkdocsMetadata.siteName);
  assert.equal(siteConfig.description, mkdocsMetadata.siteDescription);
  assert.equal(siteConfig.siteUrl, process.env.NEXT_PUBLIC_SITE_URL ?? mkdocsMetadata.siteUrl);
  assert.equal(siteConfig.repoUrl, mkdocsMetadata.repoUrl);
  assert.equal(siteConfig.copyright, mkdocsMetadata.copyright);
  assert.equal(siteConfig.remoteBranch, mkdocsMetadata.remoteBranch);
  assert.equal(siteConfig.editUri, mkdocsMetadata.editUri);
});

test("site IA labels and publication flags are sourced from mkdocs config", () => {
  const sections = readMkdocsSiteSections();

  assert.deepEqual(
    ["products", "bpftime", "projects", "others", "tutorials", "blog", "about"].map(
      (key) => sections.get(key)?.labels?.en
    ),
    ["Products", "bpftime", "Projects", "About", "Tutorial", "Blog", "About"]
  );
  assert.equal(sections.get("products")?.published?.nav, true);
  assert.equal(sections.get("bpftime")?.published?.nav, false);
  assert.equal(sections.get("GPTtrace")?.labels?.en, "AI × eBPF");
  assert.equal(sections.get("GPTtrace")?.published?.nav, true);
  assert.equal(sections.get("about")?.published?.nav, false);
  assert.equal(sections.get("about")?.published?.footerExplore, false);
  assert.equal(sections.get("GPTtrace")?.published?.footerProject, true);
  assert.equal(sections.get("wasm-bpf")?.published?.nav, false);
  assert.equal(sections.get("legacy-blog")?.published?.homeExplore, true);
  assert.equal(sections.get("legacy-blog")?.published?.footerExplore, false);
});

test("primary nav children and section sidebars are sourced from mkdocs config", () => {
  const navChildren = readMkdocsPrimaryNavChildren();
  const sidebars = readMkdocsSectionSidebars();
  const landingPages = readMkdocsSectionLandingPages();

  assert.deepEqual(
    navChildren.get("products")?.map((item) => item.href),
    ["/bpftime/", "/products/agent-runtime-infrastructure/", "/products/services/"]
  );
  assert.deepEqual(
    navChildren.get("tutorials")?.map((item) => item.href),
    [
      "/tutorials/",
      "/others/cuda-tutorial/",
      "/others/cupti-tutorial/",
      "/others/nvbit-tutorial/",
      "/tutorials/47-cuda-events/"
    ]
  );
  assert.deepEqual(
    navChildren.get("projects")?.map((item) => item.href),
    ["/bpftime/", "/eunomia-bpf/", "/wasm-bpf/", "/GPTtrace/agentsight/"]
  );
  assert.equal(navChildren.get("projects")?.[3]?.label.en, "AgentSight");
  assert.deepEqual(
    sidebars.get("projects")?.map((group) => group.title.en),
    ["Project overview", "Project docs", "Learning and writing"]
  );
  assert.equal(sidebars.get("projects")?.[1]?.items[2]?.href, "/wasm-bpf/");
  assert.ok(!sidebars.get("projects")?.some((group) => group.items.some((item) => item.href === "/blogs/")));
  assert.equal(sidebars.get("GPTtrace")?.[0]?.title.en, "AI and eBPF");
  assert.deepEqual(
    sidebars.get("GPTtrace")?.[0]?.items.map((item) => item.label.en),
    ["Overview", "AgentSight", "MCPtrace", "GPTtrace", "SchedCP"]
  );
  assert.equal(landingPages.get("projects")?.variant, "project-index");
  assert.equal(landingPages.has("GPTtrace"), false);
});

test("home project cards are sourced from mkdocs config", () => {
  const home = readMkdocsHomeConfig();

  assert.equal(home.projectsTitle.en, "Projects");
  assert.equal(home.projectsTitle.zh, "项目");
  assert.equal(home.hero.summary.en, "Open-source eBPF infrastructure for runtime extension, GPU tracing, and AI Agents.");
  assert.equal(home.capabilities.length, 3);
  assert.deepEqual(
    home.projectGroups.map((group) => group.key),
    ["platform", "resources", "ai-agents"]
  );
  assert.deepEqual(
    home.projects.map((project) => project.key),
    [
      "bpftime",
      "bpf-developer-tutorial",
      "docs",
      "blog",
      "papers",
      "eunomia-bpf",
      "GPTtrace",
      "agentsight",
      "llvmbpf",
      "wasm-bpf"
    ]
  );
  assert.ok(
    home.projects
      .find((project) => project.key === "bpftime")
      ?.links.some((link) => link.label.en === "OSDI 2025")
  );
  assert.ok(
    home.projects
      .find((project) => project.key === "GPTtrace")
      ?.links.some((link) => link.label.en === "eBPF 2024")
  );
  assert.ok(
    home.projects
      .find((project) => project.key === "agentsight")
      ?.links.some((link) => link.label.en === "arXiv 2508.02736")
  );
});

test("configured section landing copy is sourced from mkdocs config", async () => {
  const sectionPages = readMkdocsSectionPages();
  const projects = await loadSectionPage("projects", [], "en");
  const projectsZh = await loadSectionPage("projects", [], "zh");
  const products = await loadSectionPage("products", [], "en");
  const bpftime = await loadSectionPage("bpftime", [], "en");
  const agentInfra = await loadSectionPage("products", ["agent-runtime-infrastructure"], "en");
  const services = await loadSectionPage("products", ["services"], "en");
  const about = await loadSectionPage("others", [], "en");

  assert.equal(sectionPages.has("projects/index.md"), false);
  assert.equal(sectionPages.get("products/index.md")?.reactPage, "products");
  assert.equal(sectionPages.get("bpftime/index.md")?.reactPage, "bpftime-product");
  assert.deepEqual(
    sectionPages.get("products/index.md")?.links.map((link) => link.href),
    [
      "/bpftime/",
      "mailto:yusheng@eunomia.dev",
      "https://github.com/eunomia-bpf/bpftime",
      "/products/agent-runtime-infrastructure/",
      "/products/services/"
    ]
  );
  assert.equal(products?.reactPage, "products");
  assert.equal(products?.reactLinks?.find((link) => link.key === "bpftime")?.href, "/bpftime/");
  assert.equal(products?.reactLinks?.find((link) => link.key === "agent-infra")?.href, "/products/agent-runtime-infrastructure/");
  assert.equal(bpftime?.reactPage, "bpftime-product");
  assert.equal(bpftime?.reactLinks?.find((link) => link.key === "bpftime-docs")?.href, "/bpftime/documents/introduction/");
  assert.equal(agentInfra?.reactPage, "agent-runtime-infrastructure");
  assert.equal(services?.reactPage, "services");
  assert.equal(about?.reactPage, "about");
  assert.equal(about?.reactLinks?.find((link) => link.key === "cuda-tutorial")?.href, "/others/cuda-tutorial/");
  assert.equal(projects?.title, "Projects");
  assert.equal(projects?.landingPage?.variant, "project-index");
  assert.ok(projects?.projectCatalog?.projects.some((project) => project.key === "agentsight"));
  assert.ok(projects?.sidebar?.some((group) => group.title === "Project docs"));
  assert.equal(projectsZh?.title, "项目");
  assert.equal(projectsZh?.landingPage?.description.zh, "eunomia-bpf 项目体系地图，按 runtime infrastructure、开发工具链、AI agent systems 和公开资源组织。");
  assert.ok(projectsZh?.sidebar?.some((group) => group.title === "项目文档"));

  const aiEbpf = await loadSectionPage("GPTtrace", [], "en");
  assert.match(aiEbpf?.sourcePath ?? "", /docs\/GPTtrace\/index\.md$/);
  assert.equal(aiEbpf?.title, "eBPF × AI/LLMs: The Convergence of System Observability and Artificial Intelligence");
  assert.equal(aiEbpf?.sidebar?.[0]?.title, "AI and eBPF");
  assert.equal(aiEbpf?.sidebar?.[0]?.items[0]?.title, "Overview");
  assert.equal(aiEbpf?.landingPage, undefined);
  assert.ok(aiEbpf?.bodyHtml.includes("This powerful relationship is a symbiotic loop"));
});

test("blog listings use explicit article descriptions instead of repeating titles", async () => {
  const acrFenceDescription =
    "ACRFence explains semantic rollback attacks in AI agent checkpoint/restore workflows and shows how intent-aware fencing prevents duplicate irreversible actions and revived authority.";
  const acrFenceZhDescription =
    "ACRFence 介绍 AI Agent 检查点恢复中的语义回滚攻击，并说明如何用意图感知的 fencing 机制避免重复执行不可逆操作和复活已消耗授权。";
  const englishEntry = getBlogEntriesForLocale("en").find((entry) => entry.key === "agent-check-restore-safety");
  const chineseEntry = getBlogEntriesForLocale("zh").find((entry) => entry.key === "agent-check-restore-safety");
  const englishIndex = await loadBlogIndex("en");
  const chineseIndex = await loadBlogIndex("zh");
  const englishIndexEntry = englishIndex.blogEntries?.find((entry) => entry.key === "agent-check-restore-safety");
  const chineseIndexEntry = chineseIndex.blogEntries?.find((entry) => entry.key === "agent-check-restore-safety");

  assert.notEqual(englishIndex.description, englishIndex.title);
  assert.notEqual(chineseIndex.description, chineseIndex.title);

  assert.ok(englishEntry);
  assert.equal(englishEntry.description, acrFenceDescription);
  assert.notEqual(englishEntry.description, englishEntry.title);

  assert.ok(chineseEntry);
  assert.equal(chineseEntry.description, acrFenceZhDescription);
  assert.notEqual(chineseEntry.description, chineseEntry.title);

  assert.equal(englishIndexEntry?.description, acrFenceDescription);
  assert.equal(chineseIndexEntry?.description, acrFenceZhDescription);
});

test("collection page sources resolve through the family registry", () => {
  assert.equal(
    resolveCollectionPageSource("tutorial", ["1-helloworld"], "en"),
    "tutorials/1-helloworld/README.md"
  );
  assert.equal(
    resolveCollectionPageSource(
      "blog",
      ["2026", "02", "17", "agentcgroup-what-happens-when-ai-coding-agents-meet-os-resources"],
      "en"
    ),
    "blog/posts/agentcgroup-characterization.md"
  );
  assert.equal(resolveCollectionPageSource("legacy-blog", ["bpftime"], "en"), "blogs/bpftime.md");
});

test("parseMarkdown only strips a leading document H1", () => {
  const page = parseMarkdown("eunomia-bpf/index.md");

  assert.match(page.body, /# download the latest release \(aka\.pw\/bpf-ecli redirects to the current GitHub release asset\)/);
});

test("rewriteContentUrl rewrites nested relative asset paths to the static asset path", () => {
  assert.equal(
    rewriteContentUrl("./tcpconnlat1.png", "tutorials/13-tcpconnlat/README.md", "en"),
    "/_content-assets/docs/tutorials/13-tcpconnlat/tcpconnlat1.png"
  );
});

test("rewriteContentUrl rewrites same-site absolute asset paths to the static asset path", () => {
  assert.equal(
    rewriteContentUrl("https://eunomia.dev/bpftime/documents/bpftime.png", "bpftime/index.md", "en"),
    "/_content-assets/docs/bpftime/documents/bpftime.png"
  );
});

test("rewriteContentUrl resolves nested tutorial doc links to public routes", () => {
  assert.equal(
    rewriteContentUrl("../README.md", "tutorials/38-btf-uprobe/test-verify/README.md", "en"),
    "/tutorials/38-btf-uprobe/"
  );
  assert.equal(
    rewriteContentUrl("../README.zh.md", "tutorials/38-btf-uprobe/test-verify/README.zh.md", "zh"),
    "/zh/tutorials/38-btf-uprobe/"
  );
});

test("parseMarkdown extracts stable metadata from docs content", () => {
  const page = parseMarkdown("tutorials/1-helloworld/README.md");
  assert.equal(typeof page.title, "string");
  assert.notEqual(page.title.length, 0);
  assert.equal(typeof page.description, "string");
  assert.notEqual(page.description.length, 0);
});

test("getStaticAssetPublicPath generates stable public URLs", () => {
  assert.equal(
    getStaticAssetPublicPath("docs", "tutorials/13-tcpconnlat/tcpconnlat1.png"),
    "/_content-assets/docs/tutorials/13-tcpconnlat/tcpconnlat1.png"
  );
});

test("writeStaticAssets copies docs and site assets into the static asset tree", () => {
  const outputRoot = fs.mkdtempSync(path.join(os.tmpdir(), "eunomia-static-assets-"));
  const indexPath = path.join(path.dirname(outputRoot), "static-assets.json");

  try {
    const result = writeStaticAssets(outputRoot, indexPath);
    assert.ok(result.count > 0);
    assert.ok(fs.existsSync(path.join(outputRoot, "docs", "tutorials", "13-tcpconnlat", "tcpconnlat1.png")));
    assert.ok(fs.existsSync(indexPath));

    const payload = JSON.parse(fs.readFileSync(indexPath, "utf8")) as {
      assets?: Array<{ source: "docs" | "site"; relativePath: string }>;
    };
    const siteAssets = payload.assets?.filter((asset) => asset.source === "site") ?? [];
    const knownSiteAsset = path.join(siteRoot, "tutorials", "29-sockops", "merbridge.png");

    if (fs.existsSync(knownSiteAsset)) {
      assert.ok(siteAssets.some((asset) => asset.relativePath === "tutorials/29-sockops/merbridge.png"));
      assert.ok(fs.existsSync(path.join(outputRoot, "site", "tutorials", "29-sockops", "merbridge.png")));
    } else {
      assert.equal(siteAssets.length, 0);
      assert.ok(!fs.existsSync(path.join(outputRoot, "site", "tutorials", "29-sockops", "merbridge.png")));
    }
  } finally {
    fs.rmSync(outputRoot, { recursive: true, force: true });
    fs.rmSync(indexPath, { force: true });
  }
});

test("generate-static-assets succeeds when the repository site/ directory is absent", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "eunomia-static-assets-missing-site-"));
  const outputRoot = path.join(tempDir, "public-assets");
  const indexPath = path.join(tempDir, "static-assets.json");
  const missingSiteRoot = path.join(tempDir, "missing-site");

  try {
    execFileSync(
      "node",
      [
        "--import",
        "tsx",
        "-e",
        [
          "import * as assetModule from './lib/content/assets.ts';",
          `assetModule.default.writeStaticAssets(${JSON.stringify(outputRoot)}, ${JSON.stringify(indexPath)});`
        ].join(" ")
      ],
      {
        cwd: process.cwd(),
        env: {
          ...process.env,
          EUNOMIA_SITE_ROOT: missingSiteRoot
        },
        stdio: "pipe"
      }
    );

    assert.ok(fs.existsSync(path.join(outputRoot, "docs")));
    const payload = JSON.parse(fs.readFileSync(indexPath, "utf8")) as { assets?: Array<{ source: string }> };
    assert.ok(Array.isArray(payload.assets));
    assert.ok(!payload.assets?.some((asset) => asset.source === "site"));
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test("content manifest keeps localized routes for english-only section pages", () => {
  const record = getContentManifest().find((candidate) => candidate.key === "section:eunomia-bpf:setup/build");
  assert.ok(record);
  assert.equal(record.sourceByLocale.en, "eunomia-bpf/setup/build.en.md");
  assert.equal(record.sourceByLocale.zh, "eunomia-bpf/setup/build.en.md");
  assert.equal(record.routeByLocale.en, "/eunomia-bpf/setup/build/");
  assert.equal(record.routeByLocale.zh, "/zh/eunomia-bpf/setup/build/");
});

test("site IA derives sections from discovered content and keeps stable overrides", () => {
  const sections = getSiteSections();

  assert.ok(sections.some((section) => section.key === "about" && section.indexSource === "about/index.md"));
  assert.ok(sections.some((section) => section.key === "projects" && section.indexSource === "projects/index.md"));
  assert.ok(sections.some((section) => section.key === "tutorials" && section.indexSource === "tutorials/index.md"));
  assert.ok(sections.some((section) => section.key === "bpftime" && section.indexSource === "bpftime/index.md"));
  assert.ok(sections.some((section) => section.key === "wasm-bpf" && section.indexSource === "wasm-bpf/index.md"));
  assert.ok(sections.some((section) => section.key === "legacy-blog" && section.indexSource === "blogs/index.md"));
  assert.ok(sections.every((section) => section.discovered));
});

test("primary nav follows the configured external site order", () => {
  const mkdocsNavSections = readMkdocsTopLevelNavSections();

  assert.deepEqual(
    mkdocsNavSections.map((section) => section.key),
    ["tutorials", "blog", "products", "bpftime", "GPTtrace", "eunomia-bpf", "others"]
  );
  assert.deepEqual(
    getPrimaryNav("en").map((item) => item.label),
    ["Products", "Projects", "AI × eBPF", "Tutorial", "Blog", "About"]
  );
  assert.deepEqual(
    getPrimaryNav("en").map((item) => item.href),
    ["/products/", "/projects/", "/GPTtrace/", "/tutorials/", "/blog/", "/others/"]
  );
  assert.deepEqual(
    getPrimaryNav("en").find((item) => item.href === "/products/")?.children?.map((item) => item.href),
    ["/bpftime/", "/products/agent-runtime-infrastructure/", "/products/services/"]
  );
  assert.deepEqual(
    getPrimaryNav("en").find((item) => item.href === "/tutorials/")?.children?.map((item) => item.href),
    [
      "/tutorials/",
      "/others/cuda-tutorial/",
      "/others/cupti-tutorial/",
      "/others/nvbit-tutorial/",
      "/tutorials/47-cuda-events/"
    ]
  );
  assert.deepEqual(
    getPrimaryNav("en").find((item) => item.href === "/projects/")?.children?.map((item) => item.href),
    ["/bpftime/", "/eunomia-bpf/", "/wasm-bpf/", "/GPTtrace/agentsight/"]
  );
  assert.deepEqual(
    getSectionSidebarOverride("projects", "zh")?.map((group) => group.title),
    ["项目总览", "项目文档", "学习与文章"]
  );
});

test("site IA keeps non-primary published sections out of the primary nav", () => {
  const navItems = getPrimaryNav("en");

  assert.ok(!navItems.some((item) => item.href === "/blogs/"));
  assert.ok(!navItems.some((item) => item.href === "/bpftime/"));
  assert.ok(!navItems.some((item) => item.href === "/wasm-bpf/"));
  assert.ok(!navItems.some((item) => item.href === "/eunomia-bpf/"));
  assert.ok(!navItems.some((item) => item.href === "/about/"));
});

test("manifest resolves tutorial routes back to the canonical tutorial record", () => {
  const record = resolveManifestRecordFromRoute("/tutorials/1-helloworld/");

  assert.ok(record);
  assert.equal(record.kind, "tutorial-page");
  assert.equal(record.key, "tutorial:1-helloworld");
});

test("manifest resolves section index routes back to the canonical section record", () => {
  const record = resolveManifestRecordFromRoute("/bpftime/");

  assert.ok(record);
  assert.equal(record.kind, "section-page");
  assert.equal(record.key, "section:bpftime:");
});

test("manifest resolves historical Claude Code SSL blog route aliases", () => {
  const canonicalRoute = "/blog/2026/02/13/reverse-engineering-claude-codes-ssl-traffic-with-ebpf/";
  const aliasRoute = "/blog/2026/02/13/reverse-engineering-claude-code-s-ssl-traffic-with-ebpf/";
  const canonicalRecord = resolveManifestRecordFromRoute(canonicalRoute);
  const aliasRecord = resolveManifestRecordFromRoute(aliasRoute);

  assert.ok(canonicalRecord);
  assert.equal(canonicalRecord.kind, "blog-page");
  assert.equal(canonicalRecord.slug?.at(-1), "reverse-engineering-claude-codes-ssl-traffic-with-ebpf");
  assert.equal(aliasRecord?.key, canonicalRecord.key);
  assert.ok(listRenderableRoutesForLocale("en").includes(aliasRoute));
});

test("generic section route entries collapse README and index aliases into one public route", () => {
  const setupRoutes = getGenericSectionRouteEntries().filter(
    (candidate) => candidate.section === "eunomia-bpf" && candidate.slug.join("/") === "setup"
  );

  assert.equal(setupRoutes.length, 1);
  assert.ok(setupRoutes[0]?.sourceAliases.includes("eunomia-bpf/setup/index.md"));
  assert.ok(setupRoutes[0]?.sourceAliases.includes("eunomia-bpf/setup/README.md"));
});

test("content manifest keeps one setup page record and prefers the richer README source", () => {
  const setupRecords = getContentManifest().filter((candidate) => candidate.key === "section:eunomia-bpf:setup");

  assert.equal(setupRecords.length, 1);
  assert.equal(setupRecords[0]?.sourceByLocale.en, "eunomia-bpf/setup/README.en.md");
  assert.equal(setupRecords[0]?.sourceByLocale.zh, "eunomia-bpf/setup/README.en.md");
  assert.equal(setupRecords[0]?.routeByLocale.en, "/eunomia-bpf/setup/");
  assert.equal(setupRecords[0]?.routeByLocale.zh, "/zh/eunomia-bpf/setup/");
});

test("generic section routes preserve zh-only pages without inventing english routes", () => {
  const zhRoute = getGenericSectionRoutes("zh").find(
    (candidate) =>
      candidate.section === "eunomia-bpf" && candidate.slug.join("/") === "ecli/ecli-dockerfile-usage"
  );
  const enRoute = getGenericSectionRoutes("en").find(
    (candidate) =>
      candidate.section === "eunomia-bpf" && candidate.slug.join("/") === "ecli/ecli-dockerfile-usage"
  );

  assert.deepEqual(zhRoute, {
    section: "eunomia-bpf",
    slug: ["ecli", "ecli-dockerfile-usage"]
  });
  assert.equal(enRoute, undefined);
});

test("docPathToRoute maps alias section docs onto the same public route", () => {
  assert.equal(docPathToRoute("eunomia-bpf/setup/index.md", "en"), "/eunomia-bpf/setup/");
  assert.equal(docPathToRoute("eunomia-bpf/setup/index.md", "zh"), "/zh/eunomia-bpf/setup/");
  assert.equal(docPathToRoute("eunomia-bpf/setup/README.en.md", "en"), "/eunomia-bpf/setup/");
});

test("docPathToRoute preserves the only surviving localized route for zh-only section docs", () => {
  assert.equal(
    docPathToRoute("eunomia-bpf/ecli/ecli-dockerfile-usage.zh.md", "zh"),
    "/zh/eunomia-bpf/ecli/ecli-dockerfile-usage/"
  );
  assert.equal(
    docPathToRoute("eunomia-bpf/ecli/ecli-dockerfile-usage.zh.md", "en"),
    "/zh/eunomia-bpf/ecli/ecli-dockerfile-usage/"
  );
});

test("docPathToRoute keeps legacy blog routes stable across locales", () => {
  assert.equal(docPathToRoute("blogs/bpftime.md", "en"), "/blogs/bpftime/");
  assert.equal(docPathToRoute("blogs/bpftime.zh.md", "zh"), "/zh/blogs/bpftime/");
});

test("listSitemapRoutes keeps key legacy and section routes", () => {
  const rawRoutes = listSitemapRoutes();
  const routes = new Set(rawRoutes);
  assert.ok(routes.has("/zh/blogs/bpftime/"));
  assert.ok(routes.has("/eunomia-bpf/setup/build/"));
  assert.ok(routes.has("/eunomia-bpf/setup/"));
  assert.ok(routes.has("/tutorials/38-btf-uprobe/test-verify/"));
  assert.ok(routes.has("/zh/eunomia-bpf/ecli/ecli-dockerfile-usage/"));
  assert.ok(!routes.has("/eunomia-bpf/ecli/ecli-dockerfile-usage/"));
  assert.equal(routes.size, rawRoutes.length);
});

test("listSitemapRoutes respects rollout stages for dated blog cutover routes", () => {
  const shadowRoutes = new Set(listSitemapRoutes("shadow"));
  const cutoverRoutes = new Set(listSitemapRoutes("cutover"));
  const datedBlog = getBlogEntries().find((entry) => entry.date);
  assert.ok(datedBlog);
  const datedBlogRoute = `/blog/${datedBlog.year}/${datedBlog.month}/${datedBlog.day}/${datedBlog.slug}/`;

  assert.ok(shadowRoutes.has("/blog/"));
  assert.ok(!shadowRoutes.has(datedBlogRoute));
  assert.ok(cutoverRoutes.has(datedBlogRoute));
});

test("searchContent returns locale-aware tutorial results", () => {
  const enResults = searchContent("hello world", "en");
  const zhResults = searchContent("hello world", "zh");

  assert.ok(enResults.some((result) => result.href === "/tutorials/1-helloworld/"));
  assert.ok(zhResults.some((result) => result.href === "/zh/tutorials/1-helloworld/"));
});

test("searchContent indexes fallback blog content for the zh locale", () => {
  const fallbackBlogEntry = getBlogEntries().find((entry) => entry.sourceByLocale.en && !entry.sourceByLocale.zh);
  assert.ok(fallbackBlogEntry);
  const zhResults = searchContent(fallbackBlogEntry.title, "zh");
  const expectedHref = `/zh/blog/${fallbackBlogEntry.year}/${fallbackBlogEntry.month}/${fallbackBlogEntry.day}/${fallbackBlogEntry.slug}/`;

  assert.ok(zhResults.some((result) => result.href === expectedHref));
});

test("writeSearchIndexes emits public static search assets", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "eunomia-static-search-"));
  const publicSearchDir = path.join(appRoot, "public", "search-index");

  try {
    const outputs = writeSearchIndexes(tempDir);

    assert.equal(outputs.length, 2);
    assert.ok(fs.existsSync(path.join(tempDir, "en.json")));
    assert.ok(fs.existsSync(path.join(tempDir, "zh.json")));

    const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";

    execFileSync(npmCommand, ["run", "generate:search-index"], {
      cwd: appRoot,
      stdio: "ignore"
    });

    assert.ok(fs.existsSync(path.join(publicSearchDir, "en.json")));
    assert.ok(fs.existsSync(path.join(publicSearchDir, "zh.json")));

    const payload = JSON.parse(
      fs.readFileSync(path.join(publicSearchDir, "en.json"), "utf8")
    ) as { documents?: unknown[] };
    assert.ok(Array.isArray(payload.documents));
    assert.ok(payload.documents.length > 0);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test("loadSearchDocuments fails fast outside development when artifacts are missing", () => {
  const env = process.env as Record<string, string | undefined>;
  const previousNodeEnv = env.NODE_ENV;
  env.NODE_ENV = "production";

  try {
    assert.throws(
      () => loadSearchDocuments("en", { outputDir: path.join(process.cwd(), ".generated", "missing-search"), allowFallback: false }),
      /Missing prebuilt search index/
    );
  } finally {
    env.NODE_ENV = previousNodeEnv;
  }
});

test("getContentModel fails fast outside development when artifacts are missing", () => {
  const env = process.env as Record<string, string | undefined>;
  const previousNodeEnv = env.NODE_ENV;
  env.NODE_ENV = "production";

  try {
    assert.throws(
      () => getContentModel({ outputPath: path.join(process.cwd(), ".generated", "missing-content", "content-model.json"), allowFallback: false }),
      /Missing prebuilt content model/
    );
  } finally {
    env.NODE_ENV = previousNodeEnv;
  }
});

test("loadSearchDocuments can rebuild from content in development", () => {
  const env = process.env as Record<string, string | undefined>;
  const previousNodeEnv = env.NODE_ENV;
  env.NODE_ENV = "development";

  try {
    const documents = loadSearchDocuments("en", { outputDir: path.join(process.cwd(), ".generated", "missing-search-dev") });
    assert.ok(documents.length > 0);
  } finally {
    env.NODE_ENV = previousNodeEnv;
  }
});

test("article loaders expose continuation links for collection discovery", async () => {
  const tutorialPage = await loadTutorialPage(["1-helloworld"], "en");
  const blogPage = await loadBlogPage(["2026", "02", "17", "agentcgroup-what-happens-when-ai-coding-agents-meet-os-resources"], "en");

  assert.ok(tutorialPage?.continuation?.index);
  assert.equal(tutorialPage?.continuation?.index?.href, "/tutorials/");
  assert.ok(tutorialPage?.continuation?.next);
  assert.ok(blogPage?.continuation?.index);
  assert.equal(blogPage?.continuation?.index?.href, "/blog/");
});

test("collection index pages expose contextual sidebars without duplicating the global top nav", async () => {
  const page = await loadBlogIndex("en");

  assert.equal(page.title, "Blog");
  assert.equal(page.landingPage?.variant, "blog-index");
  assert.ok(page.cards && page.cards.length > 0);
  assert.ok(page.cards?.every((card) => !card.badge));
  assert.equal(page.sidebar?.length, 1);
  assert.equal(page.sidebar?.[0]?.items[0]?.href, "/blog/");
  assert.ok(page.sidebar?.[0]?.items.some((item) => item.href.includes("/blog/2026/")));
  assert.ok(!page.sidebar?.some((group) => group.items.some((item) => item.href === "/tutorials/")));
});

test("article sidebars stay contextual without reintroducing the global browse group", async () => {
  const tutorialPage = await loadTutorialPage(["1-helloworld"], "en");
  const sectionPage = await loadSectionPage("bpftime", ["llvmbpf"], "en");
  const projectsPage = await loadSectionPage("projects", [], "en");
  const searchSidebar = buildSearchSidebar("en");

  assert.ok(tutorialPage?.sidebar?.length);
  assert.ok(tutorialPage?.sidebar?.[0]?.items.some((item) => item.href === "/tutorials/1-helloworld/"));
  assert.ok(!tutorialPage?.sidebar?.some((group) => group.items.some((item) => item.href === "/blog/")));
  assert.ok(!tutorialPage?.sidebar?.some((group) => group.items.some((item) => item.href === "/blogs/")));

  assert.ok(sectionPage?.sidebar?.length);
  assert.ok(sectionPage?.sidebar?.[0]?.items.some((item) => item.href === "/bpftime/llvmbpf/"));
  assert.ok(!sectionPage?.sidebar?.some((group) => group.items.some((item) => item.href === "/tutorials/")));
  assert.ok(!projectsPage?.sidebar?.some((group) => group.items.some((item) => item.href === "/blogs/")));
  assert.ok(!searchSidebar.some((group) => group.items.some((item) => item.href === "/blogs/")));
});

test("resolveContentPage resolves manifest-backed routes without route-layer switches", async () => {
  const tutorial = await resolveContentPage("/tutorials/1-helloworld/", "en");
  const section = await resolveContentPage("/eunomia-bpf/setup/build/", "zh");

  assert.ok(tutorial);
  assert.equal(tutorial?.eyebrow, "Tutorials");
  assert.match(tutorial?.page.title ?? "", /Hello World/i);

  assert.ok(section);
  assert.equal(section?.eyebrow, "eunomia-bpf");
  assert.match(section?.page.bodyHtml ?? "", /Install Dependencies/);
});

test("createContentPageRoute enumerates manifest-backed static paths", async () => {
  const enRoute = createContentPageRoute("en");
  const zhRoute = createContentPageRoute("zh");

  const enPaths = await enRoute.getStaticPaths({});
  const zhPaths = await zhRoute.getStaticPaths({});

  assert.equal(enPaths.fallback, false);
  assert.equal(zhPaths.fallback, false);

  const englishSlugs = new Set(
    (enPaths.paths as Array<{ params?: { slug?: string[] } }>).map((entry) => `/${(entry.params?.slug ?? []).join("/")}/`)
  );
  const zhSlugs = new Set(
    (zhPaths.paths as Array<{ params?: { slug?: string[] } }>).map((entry) => `/zh/${(entry.params?.slug ?? []).join("/")}/`)
  );

  assert.ok(!englishSlugs.has("//"));
  assert.ok(!zhSlugs.has("/zh//"));
  assert.deepEqual(
    [...englishSlugs].sort(),
    listRenderableRoutesForLocale("en").filter((route) => route !== "/").sort()
  );
  assert.deepEqual(
    [...zhSlugs].sort(),
    listRenderableRoutesForLocale("zh").filter((route) => route !== "/zh/").sort()
  );
});

test("nested tutorial pages preserve deep relative links", async () => {
  const page = await loadTutorialPage(["38-btf-uprobe", "test-verify"], "en");

  assert.ok(page);
  assert.match(page.bodyHtml, /href="\/tutorials\/38-btf-uprobe\/"/);
});

test("legacy blog pages still render from the legacy blogs tree", async () => {
  const page = await loadLegacyBlogPage(["bpftime"], "en");

  assert.ok(page);
  assert.match(page.title, /bpftime: Extending eBPF from Kernel to User Space/);
});

test("section loaders render english fallback content from .en.md sources", async () => {
  const page = await loadSectionPage("eunomia-bpf", ["setup", "build"], "zh");

  assert.ok(page);
  assert.match(page.bodyHtml, /Install Dependencies/);
});

test("tutorial loaders render localized .zh.md content when it exists", async () => {
  const page = await loadTutorialPage(["1-helloworld"], "zh");

  assert.ok(page);
  assert.match(page.bodyHtml, /下载安装 eunomia-bpf 开发工具/);
});

test("bpftime continuation follows mkdocs nav order instead of file sort order", async () => {
  const page = await loadSectionPage("bpftime", ["documents", "introduction"], "en");

  assert.ok(page?.continuation?.previous);
  assert.ok(page?.continuation?.next);
  assert.equal(page?.continuation?.previous?.href, "/bpftime/");
  assert.equal(page?.continuation?.next?.href, "/bpftime/documents/build-and-test/");
});

test("english section continuation does not leak zh-only routes", async () => {
  const page = await loadSectionPage("eunomia-bpf", ["setup", "build"], "en");
  const links = [page?.continuation?.index, page?.continuation?.previous, page?.continuation?.next].filter(Boolean);

  assert.ok(links.length > 0);
  for (const link of links) {
    assert.ok(link);
    assert.doesNotMatch(link.href, /^\/zh\//);
  }
});

test("zh-only section pages only advertise the existing locale alternate", async () => {
  const page = await loadSectionPage("eunomia-bpf", ["ecli", "ecli-dockerfile-usage"], "zh");

  assert.ok(page);
  assert.deepEqual(page.alternates, {
    zh: "/zh/eunomia-bpf/ecli/ecli-dockerfile-usage/"
  });
  assert.deepEqual(
    resolveAlternatesFromDocSource("eunomia-bpf/ecli/ecli-dockerfile-usage.zh.md", "zh"),
    {
      zh: "/zh/eunomia-bpf/ecli/ecli-dockerfile-usage/"
    }
  );
});

test("renderFeed emits a stable RSS document", () => {
  const xml = renderFeed("en");

  assert.match(xml, /<rss version="2.0">/);
  assert.match(xml, /<channel>/);
  assert.match(xml, /<item>/);
  assert.match(xml, /https:\/\/eunomia\.dev\/blog\//);
});

test("static metadata generation emits feed, sitemap, robots, and shared OG assets", () => {
  const publicDir = fs.mkdtempSync(path.join(os.tmpdir(), "eunomia-static-metadata-public-"));
  const indexPath = path.join(os.tmpdir(), `eunomia-static-metadata-${Date.now()}.json`);

  try {
    const result = writeStaticMetadata({
      publicDir,
      indexPath
    });

    assert.deepEqual(result.files, ["feed.xml", path.join("zh", "feed.xml"), "sitemap.xml", "robots.txt", path.join("og", "default.svg")]);
    assert.ok(fs.existsSync(indexPath));
    assert.match(fs.readFileSync(path.join(publicDir, "feed.xml"), "utf8"), /<rss version="2.0">/);
    assert.match(fs.readFileSync(path.join(publicDir, "zh", "feed.xml"), "utf8"), /<language>zh-CN<\/language>/);
    assert.match(fs.readFileSync(path.join(publicDir, "sitemap.xml"), "utf8"), /<loc>https:\/\/eunomia\.dev\//);
    assert.match(fs.readFileSync(path.join(publicDir, "robots.txt"), "utf8"), /Sitemap: https:\/\/eunomia\.dev\/sitemap\.xml/);
    assert.match(fs.readFileSync(path.join(publicDir, "og", "default.svg"), "utf8"), /Static OG image shared by all pages/);
  } finally {
    fs.rmSync(publicDir, { recursive: true, force: true });
    fs.rmSync(indexPath, { force: true });
  }
});

test("ogImageUrl resolves to the shared static OG asset instead of a runtime API", () => {
  const ogImage = ogImageUrl();

  assert.equal(ogImage, absoluteUrl(STATIC_OG_IMAGE_PATH));
  assert.doesNotMatch(ogImage, /\/api\/og/);
});

test("searchContent includes heading-level anchor results", () => {
  const results = searchContent("Verify struct data access", "en", 24);

  assert.ok(results.some((result) => result.href.endsWith("#verify-struct-data-access")));
});

test("getGitMetadata exposes stable authors and timestamps for docs pages", () => {
  const metadata = getGitMetadata("tutorials/1-helloworld/README.md");

  assert.ok(metadata);
  assert.ok(metadata.authors.length > 0);
  assert.equal(typeof metadata.updatedAt, "string");
  assert.equal(typeof metadata.createdAt, "string");
});

test("renderMarkdown preserves allowed raw HTML used by current docs", async () => {
  const html = await renderMarkdown("GPTtrace/agentsight.md", "en");

  assert.match(html, /<div align="center">/);
  assert.match(html, /<img[^>]+src="https:\/\/github\.com\/eunomia-bpf\/agentsight\/raw\/master\/docs\/demo-tree\.png"/);
  assert.match(html, /<p><em>Real-time process tree visualization/);
});

test("renderMarkdown keeps legacy blog markdown constructs compatible", async () => {
  const html = await renderMarkdown("blogs/bpftime.md", "en");

  assert.match(html, /<table>/);
  assert.match(html, /Userspace \(ns\)/);
});

test("renderMarkdown renders footnotes used in current docs", async () => {
  const html = await renderMarkdown("eunomia-bpf/index.md", "en");

  assert.match(html, /data-footnotes/);
  assert.match(html, /data-footnote-ref/);
  assert.match(html, /data-footnote-backref/);
});

test("renderMarkdownDocument extracts TOC headings from article content", async () => {
  const rendered = await renderMarkdownDocument("tutorials/38-btf-uprobe/test-verify/README.md", "en");

  assert.ok(rendered.headings.length > 0);
  assert.deepEqual(rendered.headings[0], {
    id: "usage",
    text: "Usage",
    depth: 2
  });
  assert.ok(
    rendered.headings.some(
      (heading) => heading.id === "verify-struct-data-access" && heading.text === "Verify struct data access"
    )
  );
});

test("renderMarkdownBody applies syntax highlighting to fenced code blocks", async () => {
  const html = await renderMarkdownBody("```c\nint main() { return 0; }\n```", "tutorials/fake.md", "en");

  assert.match(html, /<figure data-rehype-pretty-code-figure="">/);
  assert.match(html, /data-language="c"/);
  assert.match(html, /<span style="color:/);
});

test("renderMarkdownBody normalizes shell and plaintext-ish aliases for highlighting", async () => {
  const shellHtml = await renderMarkdownBody("```console\nsudo ./ecli run package.json\n```", "tutorials/fake.md", "en");
  const textHtml = await renderMarkdownBody("```conf\nALLOW=true\n```", "tutorials/fake.md", "en");
  const cudaHtml = await renderMarkdownBody("```cuda\n__global__ void add() {}\n```", "tutorials/fake.md", "en");

  assert.match(shellHtml, /data-language="shellsession"/);
  assert.match(textHtml, /data-language="plaintext"/);
  assert.match(cudaHtml, /data-language="cpp"/);
});

test("renderMarkdownBody preserves mermaid fences for client-side diagram hydration", async () => {
  const html = await renderMarkdownBody("```mermaid\ngraph TD\n  A-->B\n```", "eunomia-bpf/manual.md", "en");

  assert.match(html, /<pre class="mermaid-diagram" data-mermaid-diagram="">/);
  assert.match(html, /graph TD/);
  assert.doesNotMatch(html, /data-rehype-pretty-code-figure/);
});

test("renderMarkdownBody rewrites local asset URLs inside allowed raw HTML", async () => {
  const html = await renderMarkdownBody(
    '<div align="center"><img src="./tcpconnlat1.png" alt="demo" width="800"></div>',
    "tutorials/13-tcpconnlat/README.md",
    "en"
  );

  assert.match(html, /\/_content-assets\/docs\/tutorials\/13-tcpconnlat\/tcpconnlat1\.png/);
});

test("renderMarkdownBody rewrites same-site absolute asset URLs inside allowed raw HTML", async () => {
  const html = await renderMarkdownBody(
    '<img src="https://eunomia.dev/bpftime/documents/bpftime-kernel.png" alt="kernel">',
    "bpftime/index.md",
    "en"
  );

  assert.match(html, /\/_content-assets\/docs\/bpftime\/documents\/bpftime-kernel\.png/);
  assert.doesNotMatch(html, /https:\/\/eunomia\.dev\/bpftime\/documents\/bpftime-kernel\.png/);
});

test("renderMarkdownBody strips unsafe raw HTML and dangerous URLs", async () => {
  const html = await renderMarkdownBody(
    '<script>alert(1)</script><a href="javascript:alert(1)">bad</a><img src="javascript:alert(2)" alt="x">',
    "blog/posts/fake.md",
    "en"
  );

  assert.doesNotMatch(html, /<script/i);
  assert.doesNotMatch(html, /javascript:/i);
  assert.match(html, />bad<\/a>/);
});

test("renderMarkdownBody keeps fenced HTML samples escaped", async () => {
  const html = await renderMarkdownBody("```html\n<script>alert(1)</script>\n```", "blog/posts/fake.md", "en");

  assert.match(html, /data-language="html"/);
  assert.match(html, /&#x3C;/);
  assert.match(html, /script/);
  assert.doesNotMatch(html, /<script>alert\(1\)<\/script>/);
});

test("renderMarkdownBody supports GFM tables with inline HTML line breaks", async () => {
  const html = await renderMarkdownBody(
    "| Column | Notes |\n| --- | --- |\n| Alpha | First<br>Second |",
    "blog/posts/fake.md",
    "en"
  );

  assert.match(html, /<table>/);
  assert.match(html, /<td>First<br>Second<\/td>/);
});

test("renderMarkdownBody preserves current raw HTML patterns with unquoted attributes", async () => {
  const html = await renderMarkdownBody(
    "<div align=center><img src=https://example.com/demo.png width=60% alt=demo></div>",
    "index.md",
    "en"
  );

  assert.match(html, /<div align="center">/);
  assert.match(html, /<img[^>]+src="https:\/\/example\.com\/demo\.png"/);
  assert.match(html, /width="60%"/);
});

test("assertSupportedMarkdown fails loudly for unsupported MkDocs-only directives", () => {
  assert.throws(
    () => assertSupportedMarkdown("--8<-- \"includes/snippet.md\"", "blog/posts/fake.md"),
    /Unsupported Markdown construct/
  );
  assert.throws(
    () => assertSupportedMarkdown("::: note\nbody", "blog/posts/fake.md"),
    /Unsupported Markdown construct/
  );
});

test("splitMaterialBlocks recognizes admonitions and tab groups", () => {
  const blocks = splitMaterialBlocks(`Intro paragraph.

!!! warning "Watch out"
    Danger zone.

=== "Linux"
    \`\`\`bash
    uname -a
    \`\`\`

=== "macOS"
    \`\`\`bash
    sw_vers
    \`\`\`
`);

  assert.equal(blocks.length, 3);
  assert.deepEqual(blocks[1], {
    type: "admonition",
    kind: "warning",
    title: "Watch out",
    collapsible: false,
    open: false,
    content: "Danger zone."
  });
  assert.equal(blocks[2]?.type, "tabs");
  assert.equal(blocks[2]?.items.length, 2);
  assert.equal(blocks[2]?.items[0]?.label, "Linux");
});

test("renderMarkdownBody renders admonitions and tabs into styled HTML", async () => {
  const html = await renderMarkdownBody(
    `???+ note "Why this matters"
    The docs renderer should support nested markdown.

=== "Shell"
    \`\`\`bash
    echo hello
    \`\`\`

=== "JSON"
    \`\`\`json
    {"ok":true}
    \`\`\`
`,
    "tutorials/fake.md",
    "en"
  );

  assert.match(html, /<details class="content-admonition content-admonition-note" open>/);
  assert.match(html, /<summary>Why this matters<\/summary>/);
  assert.match(html, /<div class="content-tabs">/);
  assert.match(html, /class="content-tab-label"[^>]*>Shell<\/label>/);
  assert.match(html, /class="content-tab-label"[^>]*>JSON<\/label>/);
  assert.match(html, /data-language="bash"/);
  assert.match(html, /data-language="json"/);
});
