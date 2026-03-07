import { load } from "cheerio";

import {
  baseUrl,
  requestTimeoutMs
} from "../config.mjs";

const DEFAULT_HEADERS = {
  "user-agent": "eunomia-site-audit/0.1 (+https://eunomia.dev)"
};

export function pageUrl(pathname) {
  return new URL(pathname, baseUrl).toString();
}

export function normalizeUrl(target) {
  const url = new URL(target);
  url.hash = "";
  if ((url.protocol === "https:" && url.port === "443") || (url.protocol === "http:" && url.port === "80")) {
    url.port = "";
  }
  return url.toString();
}

export function sameOrigin(target) {
  return new URL(target).origin === baseUrl.origin;
}

export function toAbsoluteUrl(value, source = baseUrl) {
  if (!value || value.startsWith("mailto:") || value.startsWith("tel:") || value.startsWith("javascript:")) {
    return null;
  }
  try {
    return normalizeUrl(new URL(value, source).toString());
  } catch {
    return null;
  }
}

export function isLikelyHtml(target) {
  const pathname = new URL(target).pathname;
  if (pathname.endsWith("/")) {
    return true;
  }
  const basename = pathname.split("/").pop() ?? "";
  if (!basename.includes(".")) {
    return true;
  }
  return /\.(html?)$/i.test(basename);
}

export async function fetchText(target, init = {}) {
  const response = await fetch(target, {
    ...init,
    headers: {
      ...DEFAULT_HEADERS,
      ...(init.headers ?? {})
    },
    signal: AbortSignal.timeout(requestTimeoutMs),
    redirect: "follow"
  });
  const text = await response.text();
  return { response, text };
}

export async function fetchStatus(target) {
  const response = await fetch(target, {
    headers: DEFAULT_HEADERS,
    signal: AbortSignal.timeout(requestTimeoutMs),
    redirect: "follow"
  });
  await response.arrayBuffer();
  return response;
}

export function parseHtml(html) {
  return load(html);
}

export function extractMeta($, selector, attribute = "content") {
  return ($(selector).attr(attribute) ?? "").trim();
}

export function printFailures(failures) {
  if (!failures.length) {
    return;
  }
  console.error("\nFailures:");
  for (const failure of failures) {
    console.error(`- ${failure}`);
  }
}
