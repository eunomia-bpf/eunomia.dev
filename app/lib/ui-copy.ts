import type { Locale } from "./site-data";

export const breadcrumbCopyByLocale = {
  en: {
    homeLabel: "Home",
    homeHref: "/",
    ariaLabel: "Breadcrumb"
  },
  zh: {
    homeLabel: "主页",
    homeHref: "/zh/",
    ariaLabel: "面包屑"
  }
} as const;

export const feedbackCopyByLocale = {
  en: {
    title: "Was this page helpful?",
    helpful: "This page was helpful",
    improve: "This page could be improved",
    thanks: "Thanks for your feedback!",
    improvePrefix: "Thanks for your feedback! Help us improve this page by using our",
    discussion: "GitHub discussion"
  },
  zh: {
    title: "这个页面有帮助吗？",
    helpful: "这个页面有帮助",
    improve: "这个页面还可以改进",
    thanks: "感谢你的反馈！",
    improvePrefix: "感谢你的反馈！欢迎通过我们的",
    discussion: "GitHub 讨论区"
  }
} as const;

export const mobileNavCopyByLocale = {
  en: {
    open: "Open navigation",
    close: "Close navigation",
    navigation: "Mobile navigation",
    sections: "Sections"
  },
  zh: {
    open: "打开导航",
    close: "关闭导航",
    navigation: "移动端导航",
    sections: "栏目"
  }
} as const;

export const pageFooterCopyByLocale = {
  en: {
    updated: "Last updated",
    created: "First published",
    authors: "Contributors",
    continue: "Continue exploring",
    backToIndex: "Back to index",
    previous: "Previous",
    next: "Next",
    edit: "Edit this page",
    shareX: "Share on X",
    shareFacebook: "Share on Facebook",
    discuss: "Join discussion",
    feed: "RSS feed",
    overflow: "more"
  },
  zh: {
    updated: "最后更新",
    created: "首次发布",
    authors: "贡献者",
    continue: "继续阅读",
    backToIndex: "返回索引",
    previous: "上一篇 / 上一页",
    next: "下一篇 / 下一页",
    edit: "编辑此页",
    shareX: "分享到 X",
    shareFacebook: "分享到 Facebook",
    discuss: "参与讨论",
    feed: "RSS 订阅",
    overflow: "更多"
  }
} as const;

export const searchBoxCopyByLocale = {
  en: {
    aria: "Search",
    placeholder: "Search docs",
    loading: "Searching",
    results: "Results",
    error: "Search failed. Open the full search page and try again.",
    empty: "No matching results.",
    viewAll: "View all results"
  },
  zh: {
    aria: "搜索",
    placeholder: "搜索文档",
    loading: "搜索中",
    results: "搜索结果",
    error: "搜索请求失败了，请打开完整搜索页重试。",
    empty: "没有找到匹配结果。",
    viewAll: "查看全部结果"
  }
} as const;

export const siteFooterCopyByLocale = {
  en: {
    explore: "Explore",
    projects: "Projects",
    community: "Community",
    copyright: "Open-source systems research, eBPF tooling, and runnable documentation from the eunomia-bpf community.",
    legacyBlog: "Legacy blog",
    sourceCode: "Site Source"
  },
  zh: {
    explore: "浏览",
    projects: "项目",
    community: "社区",
    copyright: "来自 eunomia-bpf 社区的开源系统研究、eBPF 工具和可运行文档。",
    legacyBlog: "旧博客",
    sourceCode: "网站源码"
  }
} as const;

type SearchResultsCopy = {
  eyebrow: string;
  title: string;
  intro: string;
  empty: string;
  prompt: string;
  open: string;
  share: string;
};

export function getSearchResultsCopy(locale: Locale, query: string, resultsCount: number): SearchResultsCopy {
  if (locale === "zh") {
    return {
      eyebrow: "搜索",
      title: query ? `“${query}” 的搜索结果` : "搜索站点内容",
      intro: query ? `找到 ${resultsCount} 条匹配结果。` : "输入至少 2 个字符，通过 header 搜索框检索教程、博客和文档页面。",
      empty: "没有找到匹配结果，可以换个关键词再试。",
      prompt: "输入至少 2 个字符后，再从 header 的搜索框发起搜索。",
      open: "打开结果",
      share: "分享搜索"
    };
  }

  return {
    eyebrow: "Search",
    title: query ? `Results for “${query}”` : "Search the site",
    intro: query ? `Found ${resultsCount} matching results.` : "Use at least 2 characters from the header search box to search tutorials, blog posts, and docs.",
    empty: "No matching results yet. Try a broader or more specific query.",
    prompt: "Use at least 2 characters, then search from the header search box.",
    open: "Open result",
    share: "Share search"
  };
}
