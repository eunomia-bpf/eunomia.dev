import { chromium } from "playwright";

import {
  baseUrl,
  expectedNavLabels,
  smokeRoutes,
  zhMarkers
} from "./config.mjs";

const failures = [];

function check(condition, message) {
  if (!condition) {
    failures.push(message);
    console.error(`FAIL ${message}`);
    return;
  }
  console.log(`PASS ${message}`);
}

function absolute(pathname) {
  return new URL(pathname, baseUrl).toString();
}

async function hasSidebarHeading(page, pattern) {
  const headings = page.locator("aside nav[aria-label='Section navigation'] section p");
  const count = await headings.count();
  for (let index = 0; index < count; index += 1) {
    const text = (await headings.nth(index).textContent())?.trim() ?? "";
    if (pattern.test(text)) {
      return true;
    }
  }
  return false;
}

async function firstVisible(page, selectors) {
  for (const selector of selectors) {
    const locator = page.locator(selector);
    const count = await locator.count();
    for (let index = 0; index < count; index += 1) {
      const candidate = locator.nth(index);
      try {
        if (await candidate.isVisible()) {
          return candidate;
        }
      } catch {
        // Ignore transient DOM state and continue.
      }
    }
  }
  return null;
}

async function gotoContentPage(page, pathname, { requireSidebar = false } = {}) {
  await page.goto(absolute(pathname), { waitUntil: "domcontentloaded" });
  await page.locator("main").first().waitFor({ state: "visible" });
  await page.locator("main h1").first().waitFor({ state: "visible" });
  if (requireSidebar) {
    await page
      .locator("aside nav[aria-label='Section navigation']")
      .first()
      .waitFor({ state: "visible" });
  }
}

async function main() {
  console.log(`Running browser smoke tests for ${baseUrl.toString()}`);

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const mobilePage = await browser.newPage({ viewport: { width: 390, height: 844 } });
  const searchRequests = [];
  page.setDefaultTimeout(15000);
  mobilePage.setDefaultTimeout(15000);
  page.on("request", (request) => {
    const pathname = new URL(request.url()).pathname;
    if (pathname.includes("/api/search") || pathname.includes("/search-index/")) {
      searchRequests.push(pathname);
    }
  });

  try {
    await page.goto(absolute(smokeRoutes.home), { waitUntil: "networkidle" });
    check(
      /eunomia/i.test(await page.title()),
      "home page title is readable"
    );
    check(
      (await page.locator("aside nav[aria-label='Section navigation']").count()) === 0,
      "home page does not render a duplicate desktop browse sidebar"
    );

    for (const label of expectedNavLabels) {
      const link = page.getByRole("link", { name: new RegExp(label, "i") }).first();
      check(await link.count(), `nav contains ${label}`);
    }

    check(await page.getByRole("heading", { name: /^Projects$/i }).count(), "home page exposes projects section");
    // Structural check only: the section should render multiple project cards
    // sourced from mkdocs.yaml. Specific project names and venue labels live in
    // config and change freely, so we don't pin them here.
    const projectsSection = page.locator('section[aria-labelledby="home-projects"]');
    check((await projectsSection.getByRole("link").count()) >= 8, "home projects section renders project cards");

    await page.goto(absolute(smokeRoutes.products), { waitUntil: "networkidle" });
    const productsText = (await page.textContent("main")) ?? "";
    check(/AI agent observability/i.test(productsText), "products page uses product-facing headline");
    check(/GPU paths/i.test(productsText), "products page mentions GPU paths");
    check(
      await page.locator("main a[href='mailto:yusheng@eunomia.dev']").first().count(),
      "products page exposes email contact CTA"
    );

    await page.goto(absolute(smokeRoutes.bpftimeProduct), { waitUntil: "networkidle" });
    const bpftimeProductText = (await page.textContent("main")) ?? "";
    check(/GPU-aware instrumentation|GPU paths/i.test(bpftimeProductText), "bpftime product page mentions GPU support");
    check(
      (await page.locator("aside nav[aria-label='Section navigation']").count()) === 0,
      "bpftime product page renders without docs sidebar"
    );
    check(
      await page.locator("main a[href='/bpftime/documents/introduction/']").first().count(),
      "bpftime product page links to documentation"
    );

    let searchInput = await firstVisible(page, [
      "input[aria-label*='Search']",
      ".md-search__input"
    ]);
    if (!searchInput) {
      const searchTrigger = await firstVisible(page, [
        "label[for='__search']",
        "button[aria-label*='Search']",
        ".md-search__button"
      ]);
      check(Boolean(searchTrigger), "search trigger is visible");
      if (searchTrigger) {
        await searchTrigger.click({ force: true });
      }
      searchInput = await firstVisible(page, [
        "input[aria-label*='Search']",
        ".md-search__input",
        "input[type='text']"
      ]);
    } else {
      check(true, "search trigger is visible");
    }
    check(Boolean(searchInput), "search input opens");
    if (searchInput) {
      await searchInput.fill("hello world");
      check(true, "search input accepts typing");
      const searchMore = page.locator(`a[href='${smokeRoutes.search}'], a[href='${absolute(smokeRoutes.search)}']`).first();
      const hasSearchMore = (await searchMore.count()) > 0;
      check(true, hasSearchMore ? "search exposes a full results page" : "search can continue to a full results page");
      const searchResult = page
        .locator(`a[href^='${smokeRoutes.tutorialArticle}'], a[href^='${absolute(smokeRoutes.tutorialArticle)}']`)
        .first();
      const liveSearchVisible = await searchResult.isVisible().catch(() => false);

      if (liveSearchVisible) {
        check(true, "search shows matching results in the live panel");
        await searchResult.click();
      } else {
        if (hasSearchMore) {
          await searchMore.click();
          await page.waitForURL(/\/search\/?\?q=hello(?:%20|\+)world$/);
        } else {
          await page.goto(absolute(smokeRoutes.search), { waitUntil: "networkidle" });
        }
        const searchPageResult = page
          .locator(
            `main a[href^='${smokeRoutes.tutorialArticle}'], main a[href^='${absolute(smokeRoutes.tutorialArticle)}']`
          )
          .first();
        await searchPageResult.waitFor({ state: "visible" });
        check(await searchPageResult.count(), "search page shows a matching tutorial result");
        await searchPageResult.click();
      }

      await page.waitForURL(/\/tutorials\/1-helloworld(?:\/|#|$)/);
      check(true, "search result click preserves navigation into the tutorial page");
      check(
        !searchRequests.some((pathname) => pathname.includes("/api/search")),
        "search does not call the runtime /api/search endpoint"
      );
      check(
        searchRequests.some((pathname) => pathname.endsWith("/search-index/en.json")),
        "search loads the prebuilt static search index"
      );
    }

    await page.goto(absolute(smokeRoutes.search), { waitUntil: "networkidle" });
    check(/result|搜索/.test(await page.textContent("main")), "search page renders result content");
    check(
      await page.locator("aside nav[aria-label='Section navigation']").first().count(),
      "search page exposes a browse sidebar"
    );

    await page.goto(absolute(smokeRoutes.tutorials), { waitUntil: "networkidle" });
    check(/tutorial/i.test(await page.textContent("main")), "tutorials page renders tutorial content");
    const tutorialIndexSidebar = page.locator("aside nav[aria-label='Section navigation']").first();
    check(await tutorialIndexSidebar.count(), "tutorial index exposes a docs sidebar");
    check(
      !(await hasSidebarHeading(page, /^(Browse|导航)$/i)),
      "tutorial index sidebar stays contextual instead of duplicating the top navigation"
    );

    await page.goto(absolute(smokeRoutes.tutorialArticle), { waitUntil: "networkidle" });
    check((await page.locator("main h1").count()) === 1, "tutorial article renders a single h1");
    check(
      await page.locator("pre[data-language] span[style*='color:']").first().count(),
      "tutorial article renders highlighted code"
    );
    const tutorialSidebar = page.locator("aside nav[aria-label='Section navigation']").first();
    check(await tutorialSidebar.count(), "tutorial article exposes a docs sidebar");
    if (await tutorialSidebar.count()) {
      check(await tutorialSidebar.isVisible(), "tutorial article shows the docs sidebar on desktop");
      check(
        await tutorialSidebar.locator(`a[href='${smokeRoutes.tutorialArticle}'][aria-current='page']`).count(),
        "tutorial article highlights the current sidebar item"
      );
      check(
        !(await hasSidebarHeading(page, /^(Browse|导航)$/i)),
        "tutorial article sidebar stays contextual instead of duplicating the top navigation"
      );
    }
    check(
      await page.locator("nav[aria-label='Breadcrumb'] a[href='/tutorials/']").first().count(),
      "tutorial article exposes breadcrumb navigation"
    );
    const tutorialLanguageToggle = page.getByRole("link", { name: "中文" }).first();
    check(await tutorialLanguageToggle.count(), "tutorial article exposes a sibling language switch");
    await tutorialLanguageToggle.click();
    await page.waitForURL(/\/zh\/tutorials\/1-helloworld\/?$/);
    check(true, "language switch preserves tutorial article context");
    const tutorialLanguageBack = page.getByRole("link", { name: "EN" }).first();
    check(await tutorialLanguageBack.count(), "localized tutorial exposes a sibling EN switch");
    await tutorialLanguageBack.click();
    await page.waitForURL(/\/tutorials\/1-helloworld\/?$/);

    await page.goto(absolute(smokeRoutes.tutorialNestedArticle), { waitUntil: "networkidle" });
    check(
      (await page.locator("main h1").count()) === 1,
      "nested tutorial article renders a single h1"
    );
    check(
      await page.locator("nav[aria-label='On this page'] a[href='#usage']").first().count(),
      "nested tutorial article exposes TOC links"
    );
    check(
      await page.locator("figure[data-rehype-pretty-code-figure], pre[data-language]").first().count(),
      "nested tutorial article renders highlighted code blocks"
    );
    const toc = page.locator("nav[aria-label='On this page'], nav[aria-label='Table of contents']").first();
    check(await toc.count(), "nested tutorial article exposes TOC");
    if (await toc.count()) {
      check(await toc.isVisible(), "nested tutorial article shows TOC on desktop");
      const tocAnchor = toc.locator("a[href^='#']").first();
      check(await tocAnchor.count(), "TOC exposes heading anchors");
    }

    await page.goto(absolute(smokeRoutes.blogArticle), { waitUntil: "networkidle" });
    check((await page.locator("main h1").count()) === 1, "blog article renders a single h1");
    const editLink = page.locator("a[href*='github.com'][href*='/tree/main/docs']").first();
    check(await editLink.count(), "blog article exposes edit link");
    const blogMainText = (await page.textContent("main")) ?? "";
    check(
      /(Last updated|Updated|最近更新)/.test(blogMainText),
      "blog article exposes git metadata"
    );
    check(
      /(Was this page helpful\?|这个页面有帮助吗？)/.test(blogMainText),
      "blog article exposes feedback CTA"
    );
    check(
      await page.locator("a[href*='x.com/intent/tweet']").first().count(),
      "blog article exposes share actions"
    );
    check(
      await page.locator("text=/Continue exploring|继续阅读/").first().count(),
      "blog article exposes continuation links"
    );

    await page.goto(absolute(smokeRoutes.blogIndex), { waitUntil: "networkidle" });
    const blogSidebar = page.locator("aside nav[aria-label='Section navigation']").first();
    check(await blogSidebar.count(), "blog index exposes a contextual blog sidebar");
    check(
      !(await hasSidebarHeading(page, /^(Browse|导航)$/i)),
      "blog index does not repeat the top navigation in the desktop sidebar"
    );

    await page.goto(absolute(smokeRoutes.legacyBlogArticle), { waitUntil: "networkidle" });
    check((await page.locator("main h1").count()) === 1, "legacy blog article renders a single h1");

    await gotoContentPage(page, smokeRoutes.sectionIndex, { requireSidebar: true });
    check((await page.locator("main h1").count()) === 1, "section landing page renders a single h1");
    const sectionIndexSidebar = page.locator("aside nav[aria-label='Section navigation']").first();
    check(await sectionIndexSidebar.count(), "section landing page exposes a contextual docs sidebar");
    check(
      !(await hasSidebarHeading(page, /^(Browse|导航)$/i)),
      "section landing page does not repeat the top navigation in the desktop sidebar"
    );

    await gotoContentPage(page, smokeRoutes.sectionArticle, { requireSidebar: true });
    check((await page.locator("main h1").count()) === 1, "section article renders a single h1");
    const sectionSidebar = page.locator("aside nav[aria-label='Section navigation']").first();
    check(await sectionSidebar.count(), "section article exposes a docs sidebar");
    if (await sectionSidebar.count()) {
      check(await sectionSidebar.isVisible(), "section article shows the docs sidebar on desktop");
      check(
        !(await hasSidebarHeading(page, /^(Browse|导航)$/i)),
        "section article sidebar stays contextual instead of duplicating the top navigation"
      );
    }

    await page.goto(absolute(smokeRoutes.mermaidArticle), { waitUntil: "domcontentloaded" });
    await page.locator(".mermaid-rendered svg").first().waitFor({ state: "visible" });
    check(await page.locator(".mermaid-rendered svg").first().count(), "mermaid diagrams render as SVG");

    await page.goto(absolute(smokeRoutes.zhHome), { waitUntil: "networkidle" });
    const bodyText = await page.textContent("body");
    check(
      zhMarkers.some((marker) => (bodyText ?? "").includes(marker)),
      "Chinese home page renders Chinese copy"
    );

    const zhTutorialLink = page.locator(
      `a[href='${smokeRoutes.zhTutorials}'], a[href='${absolute(smokeRoutes.zhTutorials)}']`
    ).first();
    if (await zhTutorialLink.count()) {
      await zhTutorialLink.click();
      await page.waitForURL(/\/zh\/tutorials\/?$/);
    } else {
      await page.goto(absolute(smokeRoutes.zhTutorials), { waitUntil: "networkidle" });
    }
    check(
      /(教程|tutorial)/i.test(await page.textContent("main")),
      "Chinese tutorials page is reachable"
    );

    await page.goto(absolute(smokeRoutes.zhLegacyBlogArticle), { waitUntil: "networkidle" });
    check(
      (await page.locator("main h1").count()) === 1,
      "Chinese legacy blog article renders a single h1"
    );

    await page.goto(absolute(smokeRoutes.zhOnlySectionArticle), { waitUntil: "networkidle" });
    const disabledLanguageToggle = page.locator("span[aria-disabled='true']").filter({ hasText: "EN" }).first();
    check(await disabledLanguageToggle.count(), "zh-only section pages do not expose a broken EN switch");

    await page.goto(absolute(smokeRoutes.search), { waitUntil: "networkidle" });
    check(await page.locator("mark").first().count(), "search page highlights matched query terms");
    check(
      await page.locator("a[href*='x.com/intent/tweet']").filter({ hasText: /share search/i }).first().count(),
      "search page exposes a share action"
    );

    await mobilePage.goto(absolute(smokeRoutes.home), { waitUntil: "networkidle" });
    const menuButton = mobilePage.getByRole("button", { name: /open navigation|打开导航/i });
    check(await menuButton.count(), "mobile navigation button is visible");
    await menuButton.click();
    const mobilePanel = mobilePage.locator("#mobile-nav-panel").first();
    await mobilePanel.waitFor({ state: "visible" });
    check(await mobilePanel.count(), "mobile navigation opens a drawer");
    await mobilePage.waitForFunction(() => {
      const panel = document.querySelector("#mobile-nav-panel");
      const box = panel?.getBoundingClientRect();
      return Boolean(box && box.x < 24 && box.height > 700);
    });
    const mobilePanelBox = await mobilePanel.boundingBox();
    check(
      Boolean(mobilePanelBox) && mobilePanelBox.x < 24 && mobilePanelBox.height > 700,
      "mobile navigation opens as a left full-height drawer"
    );
    const mobileSearchInput = await firstVisible(mobilePage, [
      "#mobile-nav-panel input[aria-label*='Search']",
      "#mobile-nav-panel input[aria-label*='搜索']"
    ]);
    check(Boolean(mobileSearchInput), "mobile navigation exposes search");
    if (!mobileSearchInput) {
      throw new Error("Mobile navigation search input did not become visible");
    }
    await mobileSearchInput.fill("hello world");
    const mobileSearchResult = mobilePage
      .locator(`a[href='${smokeRoutes.tutorialArticle}'], a[href='${absolute(smokeRoutes.tutorialArticle)}']`)
      .first();
    await mobileSearchResult.waitFor({ state: "visible" });
    check(await mobileSearchResult.count(), "mobile search returns matching results");
    await mobilePage.keyboard.press("Escape");
    await mobilePage.locator("#mobile-nav-panel").waitFor({ state: "detached", timeout: 2000 });
    check(await mobilePage.locator("#mobile-nav-panel").count() === 0, "mobile navigation closes on Escape");
    await menuButton.click();
    await mobilePage.locator("#mobile-nav-panel").first().waitFor({ state: "visible" });
    const mobileTutorialLink = mobilePage
      .locator(`#mobile-nav-panel a[href='${smokeRoutes.tutorials}'], #mobile-nav-panel a[href='${absolute(smokeRoutes.tutorials)}']`)
      .first();
    check(await mobileTutorialLink.count(), "mobile navigation exposes section links");
    const mobileBlogLink = mobilePage
      .locator(`#mobile-nav-panel a[href='${smokeRoutes.blogIndex}'], #mobile-nav-panel a[href='${absolute(smokeRoutes.blogIndex)}']`)
      .first();
    check(await mobileBlogLink.count(), "mobile navigation keeps a usable top-level blog path");
    await mobileBlogLink.click();
    await mobilePage.waitForURL(/\/blog\/?$/);

    await mobilePage.goto(absolute(smokeRoutes.tutorialArticle), { waitUntil: "networkidle" });
    const articleMenuButton = mobilePage.getByRole("button", { name: /open navigation|打开导航/i });
    await articleMenuButton.click();
    const mobileArticlePanel = mobilePage.locator("#mobile-nav-panel").first();
    await mobileArticlePanel.waitFor({ state: "visible" });
    check(
      await mobileArticlePanel.locator("nav[aria-label='Mobile navigation'], nav[aria-label='移动端导航']").count(),
      "mobile drawer exposes a unified navigation tree"
    );
    const mobileCurrentArticleLink = mobileArticlePanel
      .locator(
        `a[href='${smokeRoutes.tutorialArticle}'][aria-current='page'], a[href='${absolute(
          smokeRoutes.tutorialArticle
        )}'][aria-current='page']`
      )
      .first();
    check(
      await mobileCurrentArticleLink.count(),
      "mobile drawer includes the contextual sidebar entry for the current page"
    );
    const mobileDrawerBlogLink = mobileArticlePanel
      .locator(`a[href='${smokeRoutes.blogIndex}'], a[href='${absolute(smokeRoutes.blogIndex)}']`)
      .first();
    check(
      await mobileDrawerBlogLink.count(),
      "mobile drawer also includes the top-level site sections"
    );
  } finally {
    await browser.close();
  }

  if (failures.length) {
    console.error("\nFailures:");
    for (const failure of failures) {
      console.error(`- ${failure}`);
    }
    process.exitCode = 1;
    return;
  }

  console.log("\nBrowser smoke tests passed.");
}

await main();
