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

async function main() {
  console.log(`Running browser smoke tests for ${baseUrl.toString()}`);

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const mobilePage = await browser.newPage({ viewport: { width: 390, height: 844 } });
  page.setDefaultTimeout(15000);
  mobilePage.setDefaultTimeout(15000);

  try {
    await page.goto(absolute(smokeRoutes.home), { waitUntil: "networkidle" });
    check(
      /eunomia/i.test(await page.title()),
      "home page title is readable"
    );

    for (const label of expectedNavLabels) {
      const link = page.getByRole("link", { name: new RegExp(label, "i") }).first();
      check(await link.count(), `nav contains ${label}`);
    }

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
      const searchResult = page
        .locator(`a[href='${smokeRoutes.tutorialArticle}'], a[href='${absolute(smokeRoutes.tutorialArticle)}']`)
        .first();
      await searchResult.waitFor({ state: "visible" });
      check(await searchResult.count(), "search shows matching results");
      const searchMore = page.locator(`a[href='${smokeRoutes.search}'], a[href='${absolute(smokeRoutes.search)}']`).first();
      check(await searchMore.count(), "search exposes a full results page");
    }

    await page.goto(absolute(smokeRoutes.search), { waitUntil: "networkidle" });
    check(/result|搜索/.test(await page.textContent("main")), "search page renders result content");

    await page.goto(absolute(smokeRoutes.tutorials), { waitUntil: "networkidle" });
    check(/tutorial/i.test(await page.textContent("main")), "tutorials page renders tutorial content");

    const tutorialLink = page.locator(`a[href='${smokeRoutes.tutorialArticle}'], a[href='${absolute(smokeRoutes.tutorialArticle)}']`).first();
    if (await tutorialLink.count()) {
      await tutorialLink.click();
      await page.waitForURL(/\/tutorials\/1-helloworld\/?$/);
    } else {
      await page.goto(absolute(smokeRoutes.tutorialArticle), { waitUntil: "networkidle" });
    }
    check(await page.locator("main h1").first().count(), "tutorial article has h1");
    check(
      await page.locator("pre[data-language] span[style*='color:']").first().count(),
      "tutorial article renders highlighted code"
    );

    await page.goto(absolute(smokeRoutes.tutorialNestedArticle), { waitUntil: "networkidle" });
    check(
      await page.locator("main h1").first().count(),
      "nested tutorial article has h1"
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
    check(await page.locator("main h1").first().count(), "blog article has h1");
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

    await page.goto(absolute(smokeRoutes.legacyBlogArticle), { waitUntil: "networkidle" });
    check(await page.locator("main h1").first().count(), "legacy blog article has h1");

    await page.goto(absolute(smokeRoutes.sectionArticle), { waitUntil: "networkidle" });
    check(await page.locator("main h1").first().count(), "section article has h1");

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
      await page.locator("main h1").first().count(),
      "Chinese legacy blog article has h1"
    );

    await mobilePage.goto(absolute(smokeRoutes.home), { waitUntil: "networkidle" });
    const menuButton = mobilePage.getByRole("button", { name: /open navigation|打开导航/i });
    check(await menuButton.count(), "mobile navigation button is visible");
    await menuButton.click();
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
    const mobileTutorialLink = mobilePage
      .locator(`a[href='${smokeRoutes.tutorials}'], a[href='${absolute(smokeRoutes.tutorials)}']`)
      .first();
    check(await mobileTutorialLink.count(), "mobile navigation exposes section links");
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
