import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { getStaticAssetPublicPath, writeStaticAssets } from "../lib/content/assets";
import { getBlogEntries, getGenericSectionRouteEntries } from "../lib/content/collections";
import { loadHomePage } from "../lib/content/home-loader";
import { getContentModel } from "../lib/content/model";
import { renderFeed } from "../lib/content/feed";
import { getGitMetadata } from "../lib/content/git";
import { resolveAlternatesFromDocSource, resolveManifestRecordFromRoute } from "../lib/content/manifest";
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
import { resolveCollectionPageSource } from "../lib/content/registry";
import { renderMarkdown, renderMarkdownBody, renderMarkdownDocument } from "../lib/content/render";
import { docPathToRoute, getGenericSectionRoutes, listSitemapRoutes } from "../lib/content/routes";
import { loadSearchDocuments, searchContent, writeSearchIndexes } from "../lib/content/search";
import { rewriteContentUrl } from "../lib/content/rewrite";
import { siteRoot } from "../lib/content/roots";
import { resolveLocalizedSource, slugifyTitle } from "../lib/content/source";
import { absoluteUrl, ogImageUrl, STATIC_OG_IMAGE_PATH } from "../lib/seo";
import { getPrimaryNav, getSiteSections } from "../lib/site-ia";
import { createContentPageRoute } from "../lib/route-builders";
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

test("home page data is markdown-first and contains rendered home content", async () => {
  const home = await loadHomePage("en");

  assert.match(home.bodyHtml, /Unlock the Power of eBPF/);
  assert.ok(!("cards" in home));
  assert.ok(!("moreLinks" in home));
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

  assert.ok(sections.some((section) => section.key === "tutorials" && section.indexSource === "tutorials/index.md"));
  assert.ok(sections.some((section) => section.key === "bpftime" && section.indexSource === "bpftime/index.md"));
  assert.ok(sections.some((section) => section.key === "wasm-bpf" && section.indexSource === "wasm-bpf/index.md"));
  assert.ok(sections.some((section) => section.key === "legacy-blog" && section.indexSource === "blogs/index.md"));
  assert.ok(sections.every((section) => section.discovered));
});

test("primary nav is generated from the site IA registry", () => {
  assert.deepEqual(
    getPrimaryNav("en").map((item) => item.href),
    ["/tutorials/", "/blog/", "/bpftime/", "/GPTtrace/", "/eunomia-bpf/", "/others/"]
  );
});

test("site IA keeps non-primary published sections out of the primary nav", () => {
  const navItems = getPrimaryNav("en");

  assert.ok(!navItems.some((item) => item.href === "/blogs/"));
  assert.ok(!navItems.some((item) => item.href === "/wasm-bpf/"));
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
  const datedBlogRoute =
    "/blog/2026/02/17/agentcgroup-what-happens-when-ai-coding-agents-meet-os-resources/";

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
  const zhResults = searchContent("agentcgroup", "zh");

  assert.ok(
    zhResults.some(
      (result) => result.href === "/zh/blog/2026/02/17/agentcgroup-what-happens-when-ai-coding-agents-meet-os-resources/"
    )
  );
});

test("writeSearchIndexes emits public static search assets", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "eunomia-static-search-"));

  try {
    const outputs = writeSearchIndexes(tempDir);

    assert.equal(outputs.length, 2);
    assert.ok(fs.existsSync(path.join(tempDir, "en.json")));
    assert.ok(fs.existsSync(path.join(tempDir, "zh.json")));

    execFileSync(process.env.npm_execpath ?? "npm", ["run", "generate:search-index"], {
      cwd: process.cwd(),
      stdio: "ignore"
    });

    assert.ok(fs.existsSync(path.join(process.cwd(), "public", "search-index", "en.json")));
    assert.ok(fs.existsSync(path.join(process.cwd(), "public", "search-index", "zh.json")));

    const payload = JSON.parse(
      fs.readFileSync(path.join(process.cwd(), "public", "search-index", "en.json"), "utf8")
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

test("collection index pages still expose generic cards and sidebars after registry cleanup", async () => {
  const page = await loadBlogIndex("en");

  assert.ok(page.cards && page.cards.length > 0);
  assert.ok(page.cards?.every((card) => !card.badge));
  assert.equal(page.sidebar?.[1]?.items[0]?.href, "/blog/");
  assert.ok(page.sidebar?.[1]?.items.some((item) => item.href.includes("/blog/2026/")));
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
  assert.ok(englishSlugs.has("/tutorials/1-helloworld/"));
  assert.ok(englishSlugs.has("/blog/2026/02/17/agentcgroup-what-happens-when-ai-coding-agents-meet-os-resources/"));
  assert.ok(englishSlugs.has("/eunomia-bpf/setup/build/"));
  assert.ok(zhSlugs.has("/zh/tutorials/1-helloworld/"));
  assert.ok(zhSlugs.has("/zh/eunomia-bpf/setup/build/"));
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
  const ogImage = ogImageUrl("Anything", "Ignored");

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
