const PLATFORM_SELECTORS = {
  discord: {
    messageSelector: '[role="article"]',
    scrollTargetSelector: 'main [role="list"]'
  },
  slack: {
    messageSelector: '[role="document"], [role="listitem"]',
    scrollTargetSelector: '[role="list"]'
  }
};

function compactWhitespace(value) {
  return value.replace(/\s+/g, " ").trim();
}

export function normalizeVisibleMessages(messages, maxMessages = 160) {
  const seen = new Set();
  const normalized = [];

  for (const message of messages) {
    const text = compactWhitespace(String(message ?? ""));
    if (!text || seen.has(text)) {
      continue;
    }
    seen.add(text);
    normalized.push(text.slice(0, 4000));
    if (normalized.length >= maxMessages) {
      break;
    }
  }

  return normalized;
}

export async function readVisibleMessageWindow(tab, options = {}) {
  const profile = PLATFORM_SELECTORS[options.platform ?? "discord"];
  if (!profile) {
    throw new Error(`Unsupported community platform: ${options.platform}`);
  }

  const messageSelector = options.messageSelector ?? profile.messageSelector;
  const texts = await tab.playwright.locator(messageSelector).evaluateAll((elements) =>
    elements
      .filter((element) => element.getClientRects().length > 0)
      .map((element) => element.innerText || element.textContent || "")
  );

  return normalizeVisibleMessages(texts, options.maxMessages);
}

export async function reviewVisibleChannel(tab, options = {}) {
  const platform = options.platform ?? "discord";
  const profile = PLATFORM_SELECTORS[platform];
  if (!profile) {
    throw new Error(`Unsupported community platform: ${platform}`);
  }

  const maxWindows = Math.min(Math.max(options.maxWindows ?? 8, 1), 20);
  const pageKey = options.pageKey ?? "PageUp";
  const scrollTargetSelector = options.scrollTargetSelector ?? profile.scrollTargetSelector;
  const collected = [];

  for (let index = 0; index < maxWindows; index += 1) {
    collected.push(...(await readVisibleMessageWindow(tab, options)));
    if (index + 1 >= maxWindows) {
      break;
    }

    const scrollTarget = tab.playwright.locator(scrollTargetSelector).last();
    if (!(await scrollTarget.count())) {
      break;
    }
    await scrollTarget.press(pageKey);
    await tab.playwright.waitForTimeout(options.pauseMs ?? 650);
  }

  return {
    platform,
    windowsReviewed: maxWindows,
    messages: normalizeVisibleMessages(collected, options.maxMessages)
  };
}
