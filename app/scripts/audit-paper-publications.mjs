import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const siteRoot = path.resolve(scriptDir, "../..");
const englishIndexPath = path.join(siteRoot, "docs/others/papers/README.md");
const chineseIndexPath = path.join(siteRoot, "docs/others/papers/README.zh.md");
const docsRoot = path.join(siteRoot, "docs");

const args = new Set(process.argv.slice(2));
const offline = args.has("--offline");
const allowWarnings = args.has("--allow-warnings");
const reportFileArg = [...args].find((arg) => arg.startsWith("--report-file="));
const reportFile = reportFileArg ? path.resolve(reportFileArg.slice("--report-file=".length)) : null;

const errors = [];
const warnings = [];
const notes = [];

const staleReferences = [
  {
    pattern: "2509.18256",
    message: "SchedCP still references the unrelated arXiv record 2509.18256"
  },
  {
    pattern: "github.com/yunwei37/agentcgroup-paper",
    message: "AgentCgroup still references the retired personal paper repository"
  }
];

const trackedAuthor = "Yusheng Zheng";
const candidateWindowDays = 550;
const relevantTopic = /\b(eBPF|BPF|agent|agentic|scheduler|sched_ext|GPUOS|NCCL|checkpoint|verifier)\b/i;
const excludedCandidateIds = new Set([
  "2508.15980", // CXL memory infrastructure, outside the site's current research scope.
  "2604.02442" // CXL computational storage, outside the site's current research scope.
]);

function readRequiredFile(filePath) {
  if (!fs.existsSync(filePath)) {
    errors.push(`Missing required paper index: \`${path.relative(siteRoot, filePath)}\``);
    return "";
  }
  return fs.readFileSync(filePath, "utf8");
}

function listMarkdownFiles(directory) {
  const files = [];
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const absolutePath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...listMarkdownFiles(absolutePath));
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      files.push(absolutePath);
    }
  }
  return files;
}

function extractArxivIds(markdown) {
  return new Set([...markdown.matchAll(/arxiv\.org\/abs\/(\d{4}\.\d{4,5})/gi)].map((match) => match[1]));
}

function difference(left, right) {
  return [...left].filter((item) => !right.has(item));
}

function decodeXml(text) {
  return text
    .replaceAll("&amp;", "&")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'")
    .replace(/&#(\d+);/g, (_, code) => String.fromCodePoint(Number(code)));
}

function parseArxivEntries(xml) {
  return [...xml.matchAll(/<entry>([\s\S]*?)<\/entry>/g)].flatMap((entryMatch) => {
    const entry = entryMatch[1];
    const id = entry.match(/<id>https?:\/\/arxiv\.org\/abs\/([^<]+)<\/id>/)?.[1]?.replace(/v\d+$/, "");
    const title = entry.match(/<title>([\s\S]*?)<\/title>/)?.[1];
    const summary = entry.match(/<summary>([\s\S]*?)<\/summary>/)?.[1];
    const published = entry.match(/<published>([^<]+)<\/published>/)?.[1];
    if (!id || !title || !summary || !published) {
      return [];
    }
    return [{
      id,
      title: decodeXml(title).replace(/\s+/g, " ").trim(),
      summary: decodeXml(summary).replace(/\s+/g, " ").trim(),
      published
    }];
  });
}

async function fetchWithTimeout(url, timeoutMs = 20_000, headers = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      redirect: "follow",
      signal: controller.signal,
      headers: {
        "user-agent": "eunomia-paper-publication-audit/1.0",
        ...headers
      }
    });
  } finally {
    clearTimeout(timeout);
  }
}

async function mapWithConcurrency(items, limit, worker) {
  const results = new Array(items.length);
  let nextIndex = 0;
  async function run() {
    while (nextIndex < items.length) {
      const index = nextIndex;
      nextIndex += 1;
      results[index] = await worker(items[index]);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, run));
  return results;
}

async function checkExternalLinks(markdown) {
  const urls = [...new Set([...markdown.matchAll(/https:\/\/[^)\s]+/g)].map((match) => match[0]))];
  const results = await mapWithConcurrency(urls, 6, async (url) => {
    try {
      const response = await fetchWithTimeout(url);
      const acceptedProtectedStatus = [401, 403, 405, 429].includes(response.status);
      const reachable = (response.status >= 200 && response.status < 400) || acceptedProtectedStatus;
      return { url, status: response.status, reachable };
    } catch (error) {
      return { url, status: null, reachable: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  for (const result of results) {
    if (!result.reachable) {
      errors.push(`Paper index link failed: ${result.url} (${result.status ?? result.error})`);
    }
  }
  notes.push(`Checked ${results.length} external paper and artifact links.`);
}

async function checkRecentArxivCandidates(indexedIds) {
  const query = new URL("https://export.arxiv.org/api/query");
  query.searchParams.set("search_query", `au:"${trackedAuthor}"`);
  query.searchParams.set("start", "0");
  query.searchParams.set("max_results", "100");
  query.searchParams.set("sortBy", "submittedDate");
  query.searchParams.set("sortOrder", "descending");

  const response = await fetchWithTimeout(query);
  if (!response.ok) {
    errors.push(`arXiv author query failed with HTTP ${response.status}`);
    return;
  }

  const entries = parseArxivEntries(await response.text());
  const cutoff = Date.now() - candidateWindowDays * 24 * 60 * 60 * 1000;
  const candidates = entries.filter((entry) => {
    const combined = `${entry.title} ${entry.summary}`;
    return new Date(entry.published).valueOf() >= cutoff
      && relevantTopic.test(combined)
      && !excludedCandidateIds.has(entry.id);
  });

  for (const entry of candidates) {
    if (!indexedIds.has(entry.id)) {
      warnings.push(
        `Recent paper candidate is absent from the canonical index: [${entry.title}](https://arxiv.org/abs/${entry.id}) (${entry.published.slice(0, 10)})`
      );
    }
  }
  notes.push(`Compared the index with ${candidates.length} recent topic-matched arXiv records by ${trackedAuthor}.`);
}

async function checkOrganizationReadmes(indexedIds) {
  const token = process.env.GITHUB_TOKEN ?? process.env.GH_TOKEN;
  if (!token) {
    notes.push("Skipped eunomia-bpf organization README scanning because no GitHub token was available.");
    return;
  }

  const headers = {
    accept: "application/vnd.github+json",
    authorization: `Bearer ${token}`,
    "x-github-api-version": "2022-11-28"
  };
  const repositories = [];
  for (let page = 1; page <= 3; page += 1) {
    const response = await fetchWithTimeout(
      `https://api.github.com/orgs/eunomia-bpf/repos?type=public&sort=updated&per_page=100&page=${page}`,
      20_000,
      headers
    );
    if (!response.ok) {
      errors.push(`GitHub organization repository query failed with HTTP ${response.status}`);
      return;
    }
    const batch = await response.json();
    repositories.push(...batch);
    if (batch.length < 100) {
      break;
    }
  }

  const cutoff = Date.now() - candidateWindowDays * 24 * 60 * 60 * 1000;
  const activeRepositories = repositories.filter((repository) =>
    !repository.fork
    && !repository.archived
    && new Date(repository.pushed_at).valueOf() >= cutoff
  );
  const results = await mapWithConcurrency(activeRepositories, 8, async (repository) => {
    const response = await fetchWithTimeout(
      `https://api.github.com/repos/eunomia-bpf/${repository.name}/readme`,
      20_000,
      headers
    );
    if (response.status === 404) {
      return { repository: repository.name, ids: [] };
    }
    if (!response.ok) {
      return { repository: repository.name, error: `HTTP ${response.status}`, ids: [] };
    }
    const payload = await response.json();
    const readme = Buffer.from(payload.content ?? "", "base64").toString("utf8");
    const topOfReadme = readme.split("\n").slice(0, 100).join("\n");
    return { repository: repository.name, ids: [...extractArxivIds(topOfReadme)] };
  });

  for (const result of results) {
    if (result.error) {
      warnings.push(`Could not inspect eunomia-bpf/${result.repository} README (${result.error})`);
    }
    for (const id of result.ids) {
      if (!indexedIds.has(id)) {
        warnings.push(
          `Organization repository links a paper absent from the canonical index: [eunomia-bpf/${result.repository}](https://github.com/eunomia-bpf/${result.repository}) references [arXiv:${id}](https://arxiv.org/abs/${id})`
        );
      }
    }
  }
  notes.push(`Scanned the first 100 README lines in ${activeRepositories.length} recently active eunomia-bpf repositories.`);
}

function buildReport() {
  const lines = [
    "# Weekly paper publication audit",
    "",
    `Generated at ${new Date().toISOString()} from \`${path.relative(siteRoot, englishIndexPath)}\`.`,
    "",
    "## Summary",
    "",
    `- Errors: ${errors.length}`,
    `- Warnings and publication backlog: ${warnings.length}`,
    `- Checks completed: ${notes.length}`,
    ""
  ];

  if (errors.length) {
    lines.push("## Errors", "", ...errors.map((item) => `- ${item}`), "");
  }
  if (warnings.length) {
    lines.push("## Warnings and backlog", "", ...warnings.map((item) => `- ${item}`), "");
  }
  if (!errors.length && !warnings.length) {
    lines.push("## Result", "", "The paper index, links, bilingual coverage, and recent-paper scan are current.", "");
  }
  lines.push("## Check log", "", ...notes.map((item) => `- ${item}`), "");
  return `${lines.join("\n")}\n`;
}

const englishIndex = readRequiredFile(englishIndexPath);
const chineseIndex = readRequiredFile(chineseIndexPath);
const englishArxivIds = extractArxivIds(englishIndex);
const chineseArxivIds = extractArxivIds(chineseIndex);

for (const id of difference(englishArxivIds, chineseArxivIds)) {
  errors.push(`Chinese paper index is missing arXiv:${id}`);
}
for (const id of difference(chineseArxivIds, englishArxivIds)) {
  errors.push(`English paper index is missing arXiv:${id}`);
}
notes.push(`English and Chinese indexes contain ${englishArxivIds.size} matching arXiv records.`);

for (const match of englishIndex.matchAll(/^\| \[([^\]]+)]\([^\n]+\|[^\n]*(?:Blog pending|explainer pending)[^\n]*$/gim)) {
  warnings.push(`Technical blog is still pending: ${match[1]}`);
}

for (const match of englishIndex.matchAll(/\((\.\.\/\.\.\/blog\/posts\/[^)]+)\)/g)) {
  const target = path.resolve(path.dirname(englishIndexPath), match[1]);
  if (!fs.existsSync(target)) {
    errors.push(`Paper index points to a missing local blog: \`${match[1]}\``);
  }
}

const markdownFiles = fs.existsSync(docsRoot) ? listMarkdownFiles(docsRoot) : [];
for (const filePath of markdownFiles) {
  const content = fs.readFileSync(filePath, "utf8");
  for (const reference of staleReferences) {
    if (content.includes(reference.pattern)) {
      errors.push(`${reference.message}: \`${path.relative(siteRoot, filePath)}\``);
    }
  }
}
notes.push(`Scanned ${markdownFiles.length} Markdown files for stale paper IDs and repository links.`);

for (const match of englishIndex.matchAll(/\((\.\.\/\.\.\/blog\/posts\/[^)]+\.md)\)/g)) {
  const englishBlog = path.resolve(path.dirname(englishIndexPath), match[1]);
  if (!englishBlog.endsWith(".zh.md")) {
    const chineseBlog = englishBlog.replace(/\.md$/, ".zh.md");
    if (!fs.existsSync(chineseBlog)) {
      warnings.push(`Paper blog has no Chinese counterpart: \`${path.relative(siteRoot, englishBlog)}\``);
    }
  }
}

if (!offline && englishIndex) {
  await Promise.all([
    checkExternalLinks(englishIndex),
    checkRecentArxivCandidates(englishArxivIds),
    checkOrganizationReadmes(englishArxivIds)
  ]);
} else if (offline) {
  notes.push("Skipped network checks because --offline was requested.");
}

const report = buildReport();
if (reportFile) {
  fs.mkdirSync(path.dirname(reportFile), { recursive: true });
  fs.writeFileSync(reportFile, report);
}
process.stdout.write(report);

if (errors.length) {
  process.exitCode = 1;
} else if (warnings.length && !allowWarnings) {
  process.exitCode = 2;
}
