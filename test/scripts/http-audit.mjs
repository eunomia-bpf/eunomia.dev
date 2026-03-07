import { load } from "cheerio";

import { baseUrl } from "./config.mjs";
import {
  expectedLangForPath,
  extractMeta,
  fetchSitemapPaths,
  fetchText,
  pageUrl,
  printFailures,
  siblingLocalePath
} from "./lib/site.mjs";

const failures = [];

function check(condition, message) {
  if (!condition) {
    failures.push(message);
    console.error(`FAIL ${message}`);
    return false;
  }
  return true;
}

async function auditRobots() {
  const url = pageUrl("/robots.txt");
  const { response, text } = await fetchText(url);
  check(response.ok, "robots.txt is reachable");
  check(text.includes("Sitemap:"), "robots.txt advertises sitemap");
}

async function auditFeed(pathname) {
  const url = pageUrl(pathname);
  const { response, text } = await fetchText(url);
  check(response.ok, `${pathname}: feed is reachable`);
  check(
    (response.headers.get("content-type") ?? "").includes("application/rss+xml"),
    `${pathname}: feed content type is rss+xml`
  );
  check(text.includes("<rss"), `${pathname}: feed contains RSS markup`);
  check(text.includes("<item>"), `${pathname}: feed contains items`);
}

async function auditSitemap() {
  const { response, paths } = await fetchSitemapPaths();
  check(response.ok, "sitemap.xml is reachable");
  check(paths.length > 0, "sitemap contains at least one URL");
  check(paths.includes(pageUrl("/")), "sitemap includes home page");
  console.log(`Loaded ${paths.length} sitemap routes.`);
  return paths;
}

function validateSeoDocument(url, text, sitemapPathSet) {
  const $ = load(text);
  const pathname = new URL(url).pathname;
  const expectedLang = expectedLangForPath(pathname);
  const siblingLang = expectedLang === "zh" ? "en" : "zh";
  const siblingPath = siblingLocalePath(pathname);
  const canonical = $("link[rel='canonical']").attr("href") ?? "";
  const alternates = $("link[rel='alternate'][hreflang]")
    .map((_, element) => ({
      href: $(element).attr("href") ?? "",
      hreflang: $(element).attr("hreflang") ?? ""
    }))
    .get();
  const alternatesByLang = new Map(alternates.map((alternate) => [alternate.hreflang, alternate.href]));
  const rssFeed = $("link[rel='alternate'][type='application/rss+xml']").attr("href") ?? "";
  const expectedFeed = pathname.startsWith("/zh") ? pageUrl("/zh/feed.xml") : pageUrl("/feed.xml");

  check(Boolean(($("title").text() ?? "").trim()), `${pathname}: title exists`);
  check(
    extractMeta($, "meta[name='description']").length > 20,
    `${pathname}: meta description exists`
  );
  check(canonical === url, `${pathname}: canonical matches public URL`);
  check(Boolean(extractMeta($, "meta[property='og:title']")), `${pathname}: og:title exists`);
  check(
    Boolean(extractMeta($, "meta[property='og:description']")),
    `${pathname}: og:description exists`
  );
  check(Boolean(extractMeta($, "meta[property='og:image']")), `${pathname}: og:image exists`);
  check(rssFeed === expectedFeed, `${pathname}: rss alternate matches locale feed`);
  check(
    alternatesByLang.get(expectedLang) === url,
    `${pathname}: hreflang ${expectedLang} matches current URL`
  );
  const siblingHref = alternatesByLang.get(siblingLang);
  if (sitemapPathSet.has(siblingPath)) {
    check(
      siblingHref === pageUrl(siblingPath),
      `${pathname}: hreflang ${siblingLang} points to reachable sibling`
    );
  } else {
    check(
      !siblingHref,
      `${pathname}: omits missing hreflang ${siblingLang} sibling`
    );
  }
  check(
    ($("html").attr("lang") ?? "").toLowerCase() === expectedLang,
    `${pathname}: html lang is ${expectedLang}`
  );

  const html = text.toLowerCase();
  check(
    html.includes("googletagmanager.com/gtag/js") || html.includes("google-analytics.com"),
    `${pathname}: analytics script is present`
  );
}

async function auditSitemapPages(paths) {
  const sitemapPathSet = new Set(paths.map((url) => new URL(url).pathname));
  let audited = 0;
  for (const url of paths) {
    const pathname = new URL(url).pathname;
    const { response, text } = await fetchText(url);
    check(response.ok, `${pathname}: page is reachable`);
    if (response.ok) {
      validateSeoDocument(url, text, sitemapPathSet);
    }
    audited += 1;
    if (audited % 50 === 0 || audited === paths.length) {
      console.log(`Audited ${audited}/${paths.length} sitemap pages.`);
    }
  }
}

async function main() {
  console.log(`Auditing SEO and infrastructure for ${baseUrl.toString()}`);
  await auditRobots();
  await auditFeed("/feed.xml");
  await auditFeed("/zh/feed.xml");
  const paths = await auditSitemap();
  await auditSitemapPages(paths);

  if (failures.length) {
    printFailures(failures);
    process.exitCode = 1;
    return;
  }

  console.log("\nHTTP audit passed.");
}

await main();
