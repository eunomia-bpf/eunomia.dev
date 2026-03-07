import { crawlSeeds, baseUrl, maxAssets, maxPages } from "./config.mjs";
import {
  fetchStatus,
  fetchText,
  isLikelyHtml,
  normalizeUrl,
  pageUrl,
  parseHtml,
  printFailures,
  sameOrigin,
  toAbsoluteUrl
} from "./lib/site.mjs";

const failures = [];

function check(condition, message) {
  if (!condition) {
    failures.push(message);
    console.error(`FAIL ${message}`);
    return;
  }
  console.log(`PASS ${message}`);
}

async function main() {
  console.log(`Crawling internal links for ${baseUrl.toString()}`);

  const queue = crawlSeeds.map((seed) => normalizeUrl(pageUrl(seed)));
  const visitedPages = new Set();
  const checkedAssets = new Set();

  while (queue.length && visitedPages.size < maxPages) {
    const current = queue.shift();
    if (!current || visitedPages.has(current)) {
      continue;
    }

    visitedPages.add(current);
    const { response, text } = await fetchText(current);
    check(response.ok, `page reachable ${current}`);
    if (!response.ok) {
      continue;
    }

    const contentType = response.headers.get("content-type") ?? "";
    if (!contentType.includes("text/html")) {
      continue;
    }

    const $ = parseHtml(text);

    $("a[href]").each((_, element) => {
      const absolute = toAbsoluteUrl($(element).attr("href"), current);
      if (!absolute || !sameOrigin(absolute)) {
        return;
      }
      if (isLikelyHtml(absolute) && !visitedPages.has(absolute) && queue.length + visitedPages.size < maxPages) {
        queue.push(absolute);
      }
    });

    const assetSelectors = [
      ["img[src]", "src"],
      ["script[src]", "src"],
      ["source[src]", "src"],
      ["link[rel='stylesheet'][href]", "href"],
      ["link[rel='icon'][href]", "href"],
      ["link[rel='shortcut icon'][href]", "href"]
    ];

    for (const [selector, attribute] of assetSelectors) {
      $(selector).each((_, element) => {
        const absolute = toAbsoluteUrl($(element).attr(attribute), current);
        if (!absolute || !sameOrigin(absolute)) {
          return;
        }
        if (checkedAssets.size < maxAssets) {
          checkedAssets.add(absolute);
        }
      });
    }
  }

  for (const asset of checkedAssets) {
    const response = await fetchStatus(asset);
    check(response.ok, `asset reachable ${asset}`);
  }

  console.log(`Checked ${visitedPages.size} pages and ${checkedAssets.size} assets.`);

  if (failures.length) {
    printFailures(failures);
    process.exitCode = 1;
    return;
  }

  console.log("\nLink crawl passed.");
}

await main();
