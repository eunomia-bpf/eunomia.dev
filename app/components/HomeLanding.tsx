import type { HomePageData } from "../lib/page-factories";
import type { BlogEntry } from "../lib/content/types";

type HomeLandingProps = {
  page: HomePageData;
  recentPosts?: BlogEntry[];
};

const projects = [
  {
    name: "bpftime",
    tagline: "Userspace eBPF Runtime",
    description:
      "Run eBPF programs in user space with 10× faster Uprobe performance. Compatible with kernel eBPF toolchains—no code changes required.",
    github: "https://github.com/eunomia-bpf/bpftime",
    docs: "/bpftime/",
    badge: "Userspace eBPF"
  },
  {
    name: "bpf-developer-tutorial",
    tagline: "eBPF Developer Tutorial",
    description:
      "Hands-on eBPF tutorial series—from basic tracing with libbpf to advanced kernel and runtime topics. Over 40 walkthroughs with runnable examples.",
    github: "https://github.com/eunomia-bpf/bpf-developer-tutorial",
    docs: "/tutorials/",
    badge: "Tutorials"
  },
  {
    name: "llvmbpf",
    tagline: "LLVM-based eBPF VM",
    description:
      "A high-performance, multi-architecture eBPF virtual machine using LLVM JIT/AOT compilation. Standalone library with maps, relocation, and AOT deployment support.",
    github: "https://github.com/eunomia-bpf/llvmbpf",
    docs: "/blogs/userspace-ebpf/",
    badge: "Compiler"
  },
  {
    name: "wasm-bpf",
    tagline: "WebAssembly + eBPF",
    description:
      "Compile and run eBPF programs inside WebAssembly sandboxes. Brings eBPF observability to Wasm-based runtimes with a shared memory interface.",
    github: "https://github.com/eunomia-bpf/wasm-bpf",
    docs: "/wasm-bpf/",
    badge: "Wasm"
  },
  {
    name: "GPTtrace",
    tagline: "LLM-Powered eBPF Tracing",
    description:
      "Generate and execute eBPF programs with natural language. Uses large language models to translate plain-English queries into kernel tracing scripts.",
    github: "https://github.com/eunomia-bpf/GPTtrace",
    docs: "/GPTtrace/",
    badge: "AI"
  },
  {
    name: "eunomia-bpf",
    tagline: "eBPF Toolchain & SDK",
    description:
      "An all-in-one eBPF SDK featuring CO-RE support, WebAssembly integration, and a lightweight runtime. Write eBPF programs once, run everywhere.",
    github: "https://github.com/eunomia-bpf/eunomia-bpf",
    docs: "/eunomia-bpf/",
    badge: "SDK"
  }
];

function formatDate(dateStr: string | undefined): string {
  if (!dateStr) return "";
  try {
    return new Date(dateStr).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric"
    });
  } catch {
    return dateStr;
  }
}

export function HomeLanding({ recentPosts = [] }: HomeLandingProps) {
  return (
    <div className="space-y-16 pb-16">
      {/* Hero */}
      <section
        className="relative overflow-hidden rounded-xl px-6 py-16 text-white sm:px-10 sm:py-20"
        style={{
          background:
            "radial-gradient(circle at 78% 60%, rgba(255,182,41,0.50), rgba(255,182,41,0) 58%), linear-gradient(130deg, #061a33 0%, #0d305d 38%, #154173 55%, #225b95 85%)"
        }}
      >
        {/* Grid overlay */}
        <span
          aria-hidden
          className="pointer-events-none absolute inset-0 mix-blend-overlay"
          style={{
            background:
              "repeating-linear-gradient(60deg,rgba(255,255,255,0.06) 0 2px,transparent 2px 90px), repeating-linear-gradient(-15deg,rgba(255,255,255,0.04) 0 1px,transparent 1px 55px)"
          }}
        />
        <div className="relative mx-auto max-w-3xl text-center">
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-amber-300/80">
            Open Source · eBPF
          </p>
          <h1 className="mt-4 text-4xl font-bold tracking-tight sm:text-5xl">
            Unlock the Power of eBPF
          </h1>
          <p className="mt-5 text-lg leading-8 text-white/80">
            Explore and enhance eBPF with open-source tools and frameworks—from userspace runtimes
            and LLVM compilers to WebAssembly integration and LLM-powered tracing.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
            <a
              href="https://github.com/eunomia-bpf"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-lg bg-white px-5 py-2.5 text-sm font-semibold text-ink shadow transition hover:bg-slate-100"
            >
              <GitHubIcon />
              GitHub
            </a>
            <a
              href="/tutorials/"
              className="inline-flex items-center gap-2 rounded-lg border border-white/30 bg-white/10 px-5 py-2.5 text-sm font-semibold text-white backdrop-blur transition hover:bg-white/20"
            >
              View Tutorials
            </a>
            <a
              href="/blog/"
              className="inline-flex items-center gap-2 rounded-lg border border-white/30 bg-white/10 px-5 py-2.5 text-sm font-semibold text-white backdrop-blur transition hover:bg-white/20"
            >
              Read the Blog
            </a>
          </div>
        </div>
      </section>

      {/* Projects grid */}
      <section>
        <h2 className="mb-6 text-xl font-semibold tracking-tight text-ink">Projects</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {projects.map((project) => (
            <div
              key={project.name}
              className="group relative flex flex-col rounded-xl border border-slate-200 bg-white p-5 shadow-sm transition hover:shadow-panel"
            >
              <div className="mb-3 flex items-start justify-between gap-2">
                <h3 className="font-semibold tracking-tight text-ink">{project.name}</h3>
                <span className="whitespace-nowrap rounded-full bg-mist px-2.5 py-0.5 text-[11px] font-medium text-azure">
                  {project.badge}
                </span>
              </div>
              <p className="text-xs font-medium text-slate-500">{project.tagline}</p>
              <p className="mt-2 flex-1 text-sm leading-6 text-slate-600">{project.description}</p>
              <div className="mt-4 flex items-center gap-3 text-xs font-medium">
                <a
                  href={project.github}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1.5 text-slate-500 transition hover:text-ink"
                >
                  <GitHubIcon className="h-3.5 w-3.5" />
                  GitHub
                </a>
                <a
                  href={project.docs}
                  className="flex items-center gap-1 text-azure transition hover:underline"
                >
                  Docs →
                </a>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Recent blog posts */}
      {recentPosts.length > 0 && (
        <section>
          <div className="mb-6 flex items-center justify-between">
            <h2 className="text-xl font-semibold tracking-tight text-ink">Recent Posts</h2>
            <a href="/blog/" className="text-sm font-medium text-azure transition hover:underline">
              All posts →
            </a>
          </div>
          <div className="divide-y divide-slate-200 rounded-xl border border-slate-200 bg-white">
            {recentPosts.map((post) => (
              <a
                key={post.key}
                href={`/blog/${post.year}/${post.month}/${post.day}/${post.slug}/`}
                className="block px-5 py-4 transition hover:bg-slate-50"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <p className="font-medium text-ink line-clamp-1">{post.title}</p>
                    {post.description && (
                      <p className="mt-1 text-sm leading-5 text-slate-500 line-clamp-2">
                        {post.description}
                      </p>
                    )}
                  </div>
                  <time
                    dateTime={post.date}
                    className="mt-0.5 shrink-0 text-xs text-slate-400"
                  >
                    {formatDate(post.date)}
                  </time>
                </div>
              </a>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function GitHubIcon({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className}
      aria-hidden="true"
    >
      <path d="M12 2C6.477 2 2 6.484 2 12.021c0 4.428 2.865 8.185 6.839 9.51.5.092.682-.217.682-.483 0-.237-.009-.868-.013-1.703-2.782.605-3.369-1.342-3.369-1.342-.454-1.154-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.021C22 6.484 17.522 2 12 2Z" />
    </svg>
  );
}
