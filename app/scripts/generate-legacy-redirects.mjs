import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Emit static redirect stubs for legacy URLs that no longer match a current
 * route, so they 301-equivalent (meta-refresh + rel=canonical) into the live
 * canonical URL instead of returning 404. GitHub Pages has no server-side
 * redirect support, so the redirect must be a real HTML file in the export.
 *
 * Legacy classes:
 *
 *  1. `/en/<path>`  — the old i18n layout served English under `/en/`; English
 *     now lives at the site root. Mirror every emitted root page into `/en/`.
 *     (Port of the legacy MkDocs hook `hooks/en_redirects.py`, which only runs
 *     in the deprecated MkDocs build.)
 *  2. `/blog/posts/<stem>/` (+ `/zh/`, `/en/` variants) — the old blog used the
 *     source filename stem as the slug; posts now live at dated URLs
 *     `/blog/YYYY/MM/DD/<slug>/`. Map stem -> dated route via the manifest.
 *
 *  3. `/GPTtrace/agentsight/` (+ `/zh/`, `/en/` variants) — the old AgentSight
 *     docs page now lives under `/agentsight/`. This stub intentionally
 *     overwrites that exported legacy page, while the rest of `/GPTtrace/*`
 *     remains active.
 *  4. `/projects/agentsight/*` (+ `/zh/`, `/en/` variants) — an intermediate
 *     local docs path now redirects to the root-level AgentSight section.
 *     `/others/papers/*` similarly redirects to the root-level `/papers/`
 *     section after the papers library moved out of `/others/`.
 *  5. Explicit renamed blog URLs — a few public links were published with
 *     title-derived slugs before the post title changed. Map those URLs to the
 *     current canonical route by source file so future title edits do not leave
 *     stale redirects behind.
 *  6. Source/document paths that crawlers learned from old MkDocs output or
 *     unresolved relative Markdown links. These point back to the nearest
 *     canonical article/tutorial page.
 *  7. A small set of deprecated compatibility paths for old imported docs,
 *     examples, and pseudo filesystem paths that were shown in code snippets.
 *
 * Most redirect generation only ADDS URLs; the GPTtrace AgentSight class
 * overwrites that exported legacy page by design.
 */

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const appDir = path.resolve(scriptDir, "..");
const docsDir = path.resolve(appDir, "..", "docs");
const outDir = path.resolve(appDir, "out");
const manifestPath = path.join(appDir, ".generated", "content", "manifest.json");
const documentsPath = path.join(appDir, ".generated", "content", "documents.json");

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? "https://eunomia.dev").replace(/\/+$/, "");

const renamedBlogRedirects = [
  {
    source: "blog/posts/2025.md",
    paths: ["/blog/2025/02/12/ebpf-ecosystem-progress-in-20242025-a-technical-deep-dive/"]
  },
  {
    source: "blog/posts/lmp-eunomia.md",
    paths: ["/blog/2022/10/11/如何在-linux-显微镜lmp项目中开启-ebpf-之旅/"]
  },
  {
    source: "blog/posts/0_3_0-release.md",
    paths: [
      "/blog/2023/02/11/eunomia-bpf-030-release-easily-build-package-and-publish-full-ebpf-applications-by-writing-kernel-mode-code/",
      "/blog/2023/02/11/eunomia-bpf-030-发布只需编写内核态代码轻松构建打包发布完整的-ebpf-应用/"
    ]
  },
  {
    source: "blog/posts/1_0-release.md",
    paths: [
      "/blog/2024/02/11/introducing-eunomia-bpf-v10-simplifying-ebpf-with-co-re-and-webassembly/",
      "/blog/2024/02/11/eunomia-bpf-v10ebpf--wasm-质的飞跃/"
    ]
  },
  {
    source: "blog/posts/agentsight_paper.md",
    paths: [
      "/blog/2025/08/26/agentsight-让-ai-agent-的一举一动都在掌控之中基于-ebpf-的系统级可观测性方案/"
    ]
  },
  {
    source: "blog/posts/bpf-news.md",
    paths: ["/blog/2023/04/11/ebpf-进阶-内核新特性进展一览/"]
  },
  {
    source: "blog/posts/bpftime.md",
    paths: ["/blog/2023/11/11/bpftime-让-ebpf-从内核扩展到用户空间/"]
  },
  {
    source: "blog/posts/check-restore.md",
    paths: [
      "/blog/2025/05/11/checkpointrestore-systems-evolution-techniques-and-applications-in-ai-agents/"
    ]
  },
  {
    source: "blog/posts/cxlmemtest.md",
    paths: [
      "/blog/2025/06/21/the-modern-memory-testing-arsenal----a-complete-guide-to-benchmarking-tools-for-next-gen-memory-systems/"
    ]
  },
  {
    source: "blog/posts/ebpf-papers.md",
    paths: ["/blog/2024/03/11/ebpf-的演进与影响近年的关键研究论文一览/"]
  },
  {
    source: "blog/posts/ebpf-security.md",
    paths: ["/blog/2024/02/11/ebpf-运行时安全性面临的挑战与前沿创新/"]
  },
  {
    source: "blog/posts/eunomia-bpf的三月进展.md",
    paths: ["/blog/2023/03/11/eunomia-bpf-的-3-月进展/"]
  },
  {
    source: "blog/posts/GPTtrace.md",
    paths: ["/blog/2023/09/11/使用-chatgpt-通过自然语言编写-ebpf-程序和追踪-linux-系统/"]
  },
  {
    source: "blog/posts/gpu-observability-challenges.md",
    paths: ["/blog/2025/10/14/gpu可观测性差距为什么我们需要卸载到gpu设备上的ebpf/"]
  },
  {
    source: "blog/posts/gpu-profile-tool-impl.md",
    paths: ["/blog/2025/04/21/深入gpu性能分析工具现代加速器追踪工具的实现详解/"]
  },
  {
    source: "blog/posts/gpu-profile-tools-analysis.md",
    paths: ["/blog/2025/04/11/加速器工具箱gpu和其他协处理器的性能分析和追踪详解/"]
  },
  {
    source: "blog/posts/github-templates.md",
    paths: ["/blog/2023/07/11/快速构建-ebpf-项目和开发环境一键在线编译运行-ebpf-程序/"]
  },
  {
    source: "blog/posts/how-to-write-c-in-wasm.md",
    paths: ["/blog/2022/02/11/在-webassembly-中使用-cc-和-libbpf-编写-ebpf-程序/"]
  },
  {
    source: "blog/posts/iaprof-analysis.md",
    paths: ["/blog/2025/10/11/understanding-iaprof-a-deep-dive-into-aigpu-flame-graph-profiling/"]
  },
  {
    source: "blog/posts/introduce-to-wasm-bpf-bpf-community.md",
    paths: ["/blog/2023/02/11/wasm-bpf-架起-webassembly-和-ebpf-内核可编程的桥梁/"]
  },
  {
    source: "blog/posts/osdi-sosp-obser-debug.md",
    paths: [
      "/blog/2025/06/21/observability-profiling-and-debugging-in-systems-conference-20152025/",
      "/blog/2025/06/21/系统会议中的可观测性性能分析和调试20152025/"
    ]
  },
  {
    source: "blog/posts/test-for-Android.md",
    paths: ["/blog/2022/09/11/ecli-在安卓-13-上的运行测试/"]
  },
  {
    source: "blog/posts/wasm-bpf.md",
    paths: ["/blog/2023/02/11/wasm-bpf-为云原生-webassembly-提供通用的-ebpf-内核可编程能力/"]
  },
  {
    source: "blog/posts/actplane.md",
    paths: ["/blog/2026/05/31/introducing-actplane-an-ifc-policy-engine-for-ai-agent-harnesses-in-ebpf/"]
  }
];

const blogDescendantRedirects = [
  {
    source: "blog/posts/iaprof-analysis.md",
    descendants: ["docs/README.pvc.md", "src/collectors/bpf/README.md"]
  },
  {
    source: "blog/posts/nvidia-open-driver-analysis.md",
    descendants: [
      "common-analysis.md",
      "kernel-open-analysis.md",
      "nvidia-analysis.md",
      "nvidia-modeset-analysis.md"
    ]
  }
];

const deprecatedBccDocumentPaths = [
  "kernel-versions",
  "kernel-versions_en",
  "kernel_config",
  "kernel_config_en",
  "reference_guide",
  "reference_guide_en",
  "special_filtering",
  "special_filtering_en",
  "tutorial",
  "tutorial_bcc_python_developer"
];

const explicitPathRedirects = [
  ["/cdn-cgi/l/email-protection", "/"],
  ["/etc/sudoers", "/tutorials/26-sudo/"],
  ["/eu", "/"],
  ["/sys/kernel/btf/vmlinux", "/tutorials/38-btf-uprobe/"],
  ["/eunomia-bpf/support/", "/eunomia-bpf/"],
  ["/en/others/miscellaneous/about-bpftime/", "/bpftime/"],
  ["/en/tutorials/22-", "/tutorials/22-android/"],
  ["/tutorials/13-tcpconnect", "/tutorials/13-tcpconnlat/"],
  ["/tutorials/38-btf-uprobe/test-verify/verify-failed-btf/uprobe.bpf.c", "/tutorials/38-btf-uprobe/test-verify/"],
  ["/zh/tutorials/38-btf-uprobe/test-verify/verify-failed-btf/uprobe.bpf.c", "/zh/tutorials/38-btf-uprobe/test-verify/"],
  ["/zh/tutorials/scripts/guideline_basic/", "/zh/tutorials/"],
  ["/zh/tutorials/scripts/guideline_advance/", "/zh/tutorials/"],
  ["/zh/others/cuda-tutorial/basic06/", "/zh/others/cuda-tutorial/06-cnn-convolution/"]
];

function redirectHtml(target) {
  const absolute = hasExplicitProtocol(target) ? target : `${siteUrl}${target}`;
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=${target}">
  <link rel="canonical" href="${absolute}">
  <title>Redirect</title>
</head>
<body>
  <p>Redirecting to <a href="${target}">${target}</a>...</p>
</body>
</html>
`;
}

function hasExplicitProtocol(value) {
  return /^[a-z][a-z0-9+.-]*:/i.test(value);
}

function normalizeRoutePath(routePath) {
  return `/${routePath.replace(/^\/+/, "")}`;
}

/** Write a redirect stub at out/<routePath>index.html unless it already exists. */
function writeRedirect(routePath, target, options = {}) {
  const normalizedRoutePath = normalizeRoutePath(routePath);
  const relative = normalizedRoutePath.replace(/^\/+/, "");
  const existingRouteFile = path.join(outDir, relative);
  if (fs.existsSync(existingRouteFile) && fs.statSync(existingRouteFile).isFile() && !options.overwrite) {
    return false;
  }
  const dest = path.join(outDir, relative, "index.html");
  if (fs.existsSync(dest) && !options.overwrite) {
    return false;
  }
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, redirectHtml(target));
  return true;
}

function explicitLocaleForPath(routePath) {
  if (routePath === "/zh" || routePath.startsWith("/zh/")) return "zh";
  if (routePath === "/en" || routePath.startsWith("/en/")) return "en";
  return null;
}

function writeLocalizedRedirects(legacyPath, targets, options = {}) {
  const locale = explicitLocaleForPath(legacyPath);
  let count = 0;

  if (locale) {
    const target = locale === "zh" ? targets.zh ?? targets.en : targets.en ?? targets.zh;
    if (target && writeRedirect(legacyPath, target, options)) count += 1;
    return count;
  }

  const defaultTarget = targets.en ?? targets.zh;
  if (defaultTarget && writeRedirect(legacyPath, defaultTarget, options)) count += 1;
  if (targets.en && writeRedirect(`/en${legacyPath}`, targets.en, options)) count += 1;
  if (targets.zh && writeRedirect(`/zh${legacyPath}`, targets.zh, options)) count += 1;
  return count;
}

function readManifest() {
  if (!fs.existsSync(manifestPath)) {
    console.warn(`Manifest not found at ${manifestPath}`);
    return null;
  }

  return JSON.parse(fs.readFileSync(manifestPath, "utf8")).manifest;
}

function readDocumentsBySource() {
  if (!fs.existsSync(documentsPath)) {
    console.warn(`Documents index not found at ${documentsPath}; skipping title-derived redirects`);
    return new Map();
  }

  const { documents } = JSON.parse(fs.readFileSync(documentsPath, "utf8"));
  return new Map(documents.map((document) => [document.sourceRelative, document]));
}

function targetByLocale(record) {
  return {
    en: record.routeByLocale?.en,
    zh: record.routeByLocale?.zh
  };
}

function targetForLocale(record, locale) {
  return record.routeByLocale?.[locale] ?? record.routeByLocale?.en ?? record.routeByLocale?.zh ?? null;
}

function findRecordBySource(manifest, source) {
  return manifest.find(
    (candidate) =>
      candidate.sourceByLocale?.en === source ||
      candidate.sourceByLocale?.zh === source ||
      candidate.sourceAliases?.includes(source)
  );
}

function slugifyTitle(value) {
  const slug = value
    .toLowerCase()
    .normalize("NFKD")
    .replace(/\p{Mark}+/gu, "")
    .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-{2,}/g, "-");

  return slug || value.trim().replace(/\s+/g, "-");
}

function titleSlugVariants(title) {
  const base = slugifyTitle(title);
  const variants = new Set([base]);
  variants.add(base.replace(/(?<=\d)-(?=\d)/g, ""));
  variants.add(base.replace(/ai-gpu/g, "aigpu"));
  variants.add(base.replace(/checkpoint-restore/g, "checkpointrestore"));
  variants.add(base.replace(/(?<=\p{Script=Han})-(?=\p{Script=Han})/gu, ""));
  variants.add(
    base
      .replace(/(?<=\d)-(?=\d)/g, "")
      .replace(/(?<=\p{Script=Han})-(?=\p{Script=Han})/gu, "")
      .replace(/(?<=\p{Script=Han})-(?=[a-z0-9]+-\p{Script=Han})/gu, "")
      .replace(/(?<=\p{Script=Han}-[a-z0-9]+)-(?=\p{Script=Han})/gu, "")
  );
  return [...variants].filter(Boolean);
}

/** Collect every emitted page (index.html) outside the locale subtrees. */
function collectRootPages() {
  const pages = [];
  const walk = (absDir) => {
    for (const entry of fs.readdirSync(absDir, { withFileTypes: true })) {
      const abs = path.join(absDir, entry.name);
      const rel = path.relative(outDir, abs);
      const topSegment = rel.split(path.sep)[0];
      if (topSegment === "en" || topSegment === "zh") {
        continue;
      }
      if (entry.isDirectory()) {
        walk(abs);
      } else if (entry.isFile() && entry.name.endsWith(".html")) {
        pages.push(rel.split(path.sep).join("/"));
      }
    }
  };
  walk(outDir);
  return pages;
}

function generateEnMirror() {
  let count = 0;
  for (const relPath of collectRootPages()) {
    // Only mirror directory-style pages (`foo/index.html`) and the home page;
    // skip standalone files such as 404.html.
    if (relPath !== "index.html" && !relPath.endsWith("/index.html")) {
      continue;
    }
    // Root-relative URL this page serves: index.html -> directory URL.
    const target = `/${relPath}`.replace(/(^|\/)index\.html$/, "$1");
    if (writeRedirect(`/en/${target.replace(/^\//, "")}`, target)) {
      count += 1;
    }
  }
  return count;
}

function generateLegacyBlogRedirects() {
  const manifest = readManifest();
  if (!manifest) {
    console.warn("Skipping legacy blog redirects");
    return 0;
  }

  let count = 0;

  for (const record of manifest) {
    const source = record.sourceByLocale?.en ?? record.sourceByLocale?.zh;
    if (!source || !/^blog\/posts\//.test(source)) {
      continue;
    }
    const stem = source
      .replace(/^blog\/posts\//, "")
      .replace(/\.(zh|en)\.md$/, "")
      .replace(/\.md$/, "");

    const enTarget = record.routeByLocale?.en;
    const zhTarget = record.routeByLocale?.zh;

    if (enTarget) {
      if (writeRedirect(`/blog/posts/${stem}/`, enTarget)) count += 1;
      if (writeRedirect(`/en/blog/posts/${stem}/`, enTarget)) count += 1;
    }
    if (zhTarget) {
      if (writeRedirect(`/zh/blog/posts/${stem}/`, zhTarget)) count += 1;
    }
  }
  return count;
}

function generateTitleDerivedBlogRedirects() {
  const manifest = readManifest();
  if (!manifest) {
    console.warn("Skipping title-derived blog redirects");
    return 0;
  }

  const documentsBySource = readDocumentsBySource();
  let count = 0;

  for (const record of manifest) {
    if (record.kind !== "blog-page") {
      continue;
    }

    const [year, month, day, canonicalSlug] = record.slug ?? [];
    if (!year || !month || !day) {
      continue;
    }

    const titles = Object.values(record.sourceByLocale ?? {})
      .map((source) => documentsBySource.get(source)?.title)
      .filter(Boolean);

    for (const title of titles) {
      for (const slug of titleSlugVariants(title)) {
        if (slug === canonicalSlug) {
          continue;
        }
        count += writeLocalizedRedirects(`/blog/${year}/${month}/${day}/${slug}/`, targetByLocale(record));
      }
    }
  }

  return count;
}

function generateRenamedBlogRedirects() {
  const manifest = readManifest();
  if (!manifest) {
    console.warn("Skipping renamed blog redirects");
    return 0;
  }

  let count = 0;

  for (const redirect of renamedBlogRedirects) {
    const record = findRecordBySource(manifest, redirect.source);

    if (!record) {
      console.warn(`No manifest record found for renamed blog source ${redirect.source}`);
      continue;
    }

    for (const legacyPath of redirect.paths) {
      count += writeLocalizedRedirects(legacyPath, targetByLocale(record), { overwrite: true });
    }
  }

  return count;
}

function generateBlogDescendantRedirects() {
  const manifest = readManifest();
  if (!manifest) {
    console.warn("Skipping blog descendant redirects");
    return 0;
  }

  let count = 0;

  for (const redirect of blogDescendantRedirects) {
    const record = findRecordBySource(manifest, redirect.source);
    if (!record) {
      console.warn(`No manifest record found for blog descendant source ${redirect.source}`);
      continue;
    }

    for (const descendant of redirect.descendants) {
      for (const locale of ["en", "zh"]) {
        const target = targetForLocale(record, locale);
        if (!target) {
          continue;
        }
        if (writeRedirect(`${target}${descendant}`, target)) count += 1;
      }

      const stems = new Set(
        Object.values(record.sourceByLocale ?? {})
          .filter(Boolean)
          .map((source) =>
            path.posix
              .basename(source)
              .replace(/\.(zh|zh-CN|en)\.md$/, "")
              .replace(/\.md$/, "")
          )
      );

      for (const stem of stems) {
        count += writeLocalizedRedirects(`/blog/posts/${stem}/${descendant}`, targetByLocale(record));
      }
    }
  }

  return count;
}

function generateLegacyGpttraceRedirects() {
  const legacyPaths = ["/GPTtrace/agentsight/"];
  let count = 0;

  for (const legacyPath of legacyPaths) {
    if (writeRedirect(legacyPath, "/agentsight/", { overwrite: true })) count += 1;
    if (writeRedirect(`/en${legacyPath}`, "/agentsight/", { overwrite: true })) count += 1;
    if (writeRedirect(`/zh${legacyPath}`, "/zh/agentsight/", { overwrite: true })) count += 1;
  }

  return count;
}

function generateMovedPapersRedirects() {
  const movedPaths = [
    ["others/papers", "papers"],
    ["others/papers/osdi20-brunella", "papers/osdi20-brunella"]
  ];
  let count = 0;

  for (const [from, to] of movedPaths) {
    if (writeRedirect(`/${from}/`, `/${to}/`)) count += 1;
    if (writeRedirect(`/en/${from}/`, `/${to}/`)) count += 1;
    if (writeRedirect(`/zh/${from}/`, `/zh/${to}/`)) count += 1;
  }

  return count;
}

function generateMovedAgentsightRedirects() {
  const movedPaths = [
    ["projects/agentsight", "agentsight"],
    ["projects/agentsight/quickstart", "agentsight/quickstart"],
    ["projects/agentsight/architecture", "agentsight/architecture"],
    ["projects/agentsight/visualization", "agentsight/visualization"],
    ["projects/agentsight/operational-notes", "agentsight/operational-notes"],
    ["projects/agentsight/troubleshooting", "agentsight/troubleshooting"]
  ];
  let count = 0;

  for (const [from, to] of movedPaths) {
    if (writeRedirect(`/${from}/`, `/${to}/`)) count += 1;
    if (writeRedirect(`/en/${from}/`, `/${to}/`)) count += 1;
    if (writeRedirect(`/zh/${from}/`, `/zh/${to}/`)) count += 1;
  }

  return count;
}

function sourcePathVariants(sourceRelative) {
  const variants = new Set([`/${sourceRelative}`]);

  if (sourceRelative.endsWith(".md")) {
    variants.add(`/${sourceRelative.replace(/\.md$/, "")}/`);
  }

  if (sourceRelative.startsWith("tutorials/")) {
    const tutorialSrcPath = `tutorials/src/${sourceRelative.replace(/^tutorials\//, "")}`;
    variants.add(`/${tutorialSrcPath}`);
    if (tutorialSrcPath.endsWith(".md")) {
      variants.add(`/${tutorialSrcPath.replace(/\.md$/, "")}/`);
    }
  }

  return variants;
}

function generateManifestSourceRedirects() {
  const manifest = readManifest();
  if (!manifest) {
    console.warn("Skipping manifest source redirects");
    return 0;
  }

  let count = 0;

  for (const record of manifest) {
    for (const locale of ["en", "zh"]) {
      const target = targetForLocale(record, locale);
      if (!target) {
        continue;
      }

      const sources = new Set([
        ...Object.values(record.sourceByLocale ?? {}).filter(Boolean),
        ...(record.sourceAliases ?? [])
      ]);

      for (const source of sources) {
        for (const sourcePath of sourcePathVariants(source)) {
          if (locale === "en") {
            if (writeRedirect(sourcePath, target)) count += 1;
            if (writeRedirect(`/en${sourcePath}`, target)) count += 1;
          } else {
            if (writeRedirect(`/zh${sourcePath}`, target)) count += 1;
            if (/(\.zh|\.zh-CN)(\.md)?\/?$/.test(sourcePath)) {
              if (writeRedirect(sourcePath, target)) count += 1;
              if (writeRedirect(`/en${sourcePath}`, target)) count += 1;
            }
          }
        }
      }
    }
  }

  return count;
}

function walkEntries(root, visitor) {
  if (!fs.existsSync(root)) {
    return;
  }

  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const absolute = path.join(root, entry.name);
    visitor(absolute, entry);
    if (entry.isDirectory()) {
      walkEntries(absolute, visitor);
    }
  }
}

function sourcePrefixFor(sourceRelative) {
  if (sourceRelative.endsWith("/README.md") || sourceRelative.endsWith("/index.md")) {
    return path.posix.dirname(sourceRelative);
  }
  return sourceRelative.replace(/\.md$/, "");
}

function buildTutorialPrefixTargets(manifest) {
  const targets = {
    en: [],
    zh: []
  };

  for (const record of manifest) {
    if (record.kind !== "tutorial-page") {
      continue;
    }

    for (const locale of ["en", "zh"]) {
      const target = targetForLocale(record, locale);
      if (!target) {
        continue;
      }

      const sources = new Set([
        ...Object.values(record.sourceByLocale ?? {}).filter(Boolean),
        ...(record.sourceAliases ?? [])
      ]);

      for (const source of sources) {
        targets[locale].push({
          prefix: sourcePrefixFor(source),
          target
        });
      }
    }
  }

  for (const locale of ["en", "zh"]) {
    targets[locale].sort((left, right) => right.prefix.length - left.prefix.length);
  }

  return targets;
}

function nearestTutorialTarget(relativePath, prefixTargets) {
  const normalized = relativePath.replace(/\/+$/, "");
  const sourceLike = normalized.startsWith("tutorials/src/")
    ? `tutorials/${normalized.replace(/^tutorials\/src\//, "")}`
    : normalized;

  for (const { prefix, target } of prefixTargets) {
    if (sourceLike === prefix || sourceLike.startsWith(`${prefix}/`)) {
      return target;
    }
  }

  return null;
}

function tutorialTreePathVariants(relativePath, isDirectory) {
  const normalizedPath = isDirectory ? `${relativePath}/` : relativePath;
  const variants = new Set([`/${normalizedPath}`]);

  if (!isDirectory) {
    const extension = path.posix.extname(relativePath);
    if (extension) {
      variants.add(`/${relativePath.slice(0, -extension.length)}`);
    }
    variants.add(`/_content-assets/docs/${relativePath}`);
  }

  if (relativePath.startsWith("tutorials/")) {
    const srcRelative = `tutorials/src/${relativePath.replace(/^tutorials\//, "")}${isDirectory ? "/" : ""}`;
    variants.add(`/${srcRelative}`);
    if (!isDirectory) {
      const extension = path.posix.extname(srcRelative);
      if (extension) {
        variants.add(`/${srcRelative.slice(0, -extension.length)}`);
      }
    }
  }

  return variants;
}

function hasHiddenPathSegment(relativePath) {
  return relativePath.split("/").some((segment) => segment.startsWith("."));
}

function generateTutorialTreeRedirects() {
  const manifest = readManifest();
  if (!manifest) {
    console.warn("Skipping tutorial tree redirects");
    return 0;
  }

  const prefixTargets = buildTutorialPrefixTargets(manifest);
  let count = 0;

  walkEntries(path.join(docsDir, "tutorials"), (absolute, entry) => {
    const relativePath = path.relative(docsDir, absolute).split(path.sep).join("/");
    if (hasHiddenPathSegment(relativePath)) {
      return;
    }

    const isDirectory = entry.isDirectory();

    for (const locale of ["en", "zh"]) {
      const target = nearestTutorialTarget(relativePath, prefixTargets[locale]);
      if (!target) {
        continue;
      }

      for (const routePath of tutorialTreePathVariants(relativePath, isDirectory)) {
        if (locale === "en") {
          if (writeRedirect(routePath, target)) count += 1;
          if (writeRedirect(`/en${routePath}`, target)) count += 1;
        } else {
          if (writeRedirect(`/zh${routePath}`, target)) count += 1;
        }
      }
    }
  });

  return count;
}

function generateDeprecatedTutorialRedirects() {
  let count = 0;

  for (const bccPath of deprecatedBccDocumentPaths) {
    count += writeLocalizedRedirects(`/tutorials/bcc-documents/${bccPath}/`, {
      en: "/tutorials/",
      zh: "/zh/tutorials/"
    });
  }

  return count;
}

function generateExplicitPathRedirects() {
  let count = 0;

  for (const [from, to] of explicitPathRedirects) {
    if (writeRedirect(from, to)) count += 1;
  }

  return count;
}

if (!fs.existsSync(outDir)) {
  console.error(`No static export found at ${outDir}`);
  process.exit(1);
}

// Order matters: broad compatibility rules write only missing stubs first.
// Explicit renamed blog and GPTtrace rules then overwrite stale pages where the
// old route may already exist. Finally, the /en/ mirror fills in every other
// page without clobbering existing stubs, avoiding chains.
const blogCount = generateLegacyBlogRedirects();
const titleBlogCount = generateTitleDerivedBlogRedirects();
const blogDescendantCount = generateBlogDescendantRedirects();
const sourcePathCount = generateManifestSourceRedirects();
const tutorialTreeCount = generateTutorialTreeRedirects();
const deprecatedTutorialCount = generateDeprecatedTutorialRedirects();
const explicitPathCount = generateExplicitPathRedirects();
const gpttraceCount = generateLegacyGpttraceRedirects();
const movedAgentsightCount = generateMovedAgentsightRedirects();
const movedPapersCount = generateMovedPapersRedirects();
const renamedBlogCount = generateRenamedBlogRedirects();
const enCount = generateEnMirror();

console.log(
  `Generated ${blogCount} legacy blog redirects, ${titleBlogCount} title-derived blog redirects, ` +
    `${blogDescendantCount} blog descendant redirects, ${sourcePathCount} source path redirects, ` +
    `${tutorialTreeCount} tutorial tree redirects, ${deprecatedTutorialCount} deprecated tutorial redirects, ` +
    `${explicitPathCount} explicit path redirects, ${gpttraceCount} GPTtrace redirects, ` +
    `${movedAgentsightCount} moved AgentSight redirects, ${movedPapersCount} moved papers redirects, ` +
    `${renamedBlogCount} renamed blog redirects, ` +
    `and ${enCount} /en/ redirects in ${outDir}`
);
