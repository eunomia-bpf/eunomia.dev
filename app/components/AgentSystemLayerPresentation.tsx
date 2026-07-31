import Image from "next/image";
import { useCallback, useEffect, useState } from "react";

import type { ReactPageLink } from "../lib/content/types";

type PresentationProps = {
  links: ReactPageLink[];
};

const slides = [
  { id: "opening", label: "Opening" },
  { id: "semantic-gap", label: "Semantic gap" },
  { id: "long-running", label: "Long-running agents" },
  { id: "observe", label: "Observe" },
  { id: "enforce", label: "Enforce" },
  { id: "policy", label: "Policy primitives" },
  { id: "continuity", label: "Continuity" },
  { id: "vision", label: "Vision" }
] as const;

const imageBase = "/_content-assets/docs";

function slideNumber(index: number) {
  return `${String(index + 1).padStart(2, "0")} / ${String(slides.length).padStart(2, "0")}`;
}

function Slide({
  id,
  index,
  className,
  children
}: {
  id: string;
  index: number;
  className: string;
  children: React.ReactNode;
}) {
  return (
    <section
      id={id}
      data-presentation-slide
      className={`relative min-h-[calc(100svh-8rem)] scroll-mt-20 border-b border-slate-200 px-5 py-14 sm:px-10 lg:px-14 lg:py-16 ${className}`}
    >
      <p className="relative z-10 text-xs font-semibold text-current opacity-55">{slideNumber(index)}</p>
      {children}
    </section>
  );
}

function ProjectLink({ link, accent }: { link?: ReactPageLink; accent: string }) {
  if (!link) {
    return null;
  }

  return (
    <a
      href={link.href}
      target="_blank"
      rel="noopener"
      className={`inline-flex min-h-11 items-center rounded-md border px-4 py-2 text-sm font-semibold transition ${accent}`}
    >
      {link.label}
      <span aria-hidden="true" className="ml-2">
        ↗
      </span>
    </a>
  );
}

function Screenshot({
  src,
  alt,
  caption,
  className = ""
}: {
  src: string;
  alt: string;
  caption: string;
  className?: string;
}) {
  return (
    <figure className={`overflow-hidden rounded-md border border-slate-200 bg-white ${className}`}>
      <div className="relative aspect-[16/9] bg-slate-50">
        <Image src={src} alt={alt} fill sizes="(min-width: 1024px) 50vw, 100vw" className="object-contain" unoptimized />
      </div>
      <figcaption className="border-t border-slate-200 px-4 py-3 text-sm leading-6 text-slate-600">{caption}</figcaption>
    </figure>
  );
}

export function AgentSystemLayerPresentation({ links }: PresentationProps) {
  const [activeSlide, setActiveSlide] = useState(0);
  const linkByKey = new Map(links.map((link) => [link.key, link]));

  const goToSlide = useCallback((index: number) => {
    const boundedIndex = Math.min(Math.max(index, 0), slides.length - 1);
    document.getElementById(slides[boundedIndex].id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  useEffect(() => {
    const sections = slides
      .map((slide) => document.getElementById(slide.id))
      .filter((section): section is HTMLElement => Boolean(section));
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((left, right) => right.intersectionRatio - left.intersectionRatio)[0];
        if (!visible) {
          return;
        }
        const index = slides.findIndex((slide) => slide.id === visible.target.id);
        if (index >= 0) {
          setActiveSlide(index);
        }
      },
      { rootMargin: "-18% 0px -52%", threshold: [0.1, 0.35, 0.6] }
    );

    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }
      if (event.key === "ArrowDown" || event.key === "PageDown") {
        event.preventDefault();
        goToSlide(activeSlide + 1);
      } else if (event.key === "ArrowUp" || event.key === "PageUp") {
        event.preventDefault();
        goToSlide(activeSlide - 1);
      } else if (event.key === "Home") {
        event.preventDefault();
        goToSlide(0);
      } else if (event.key === "End") {
        event.preventDefault();
        goToSlide(slides.length - 1);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeSlide, goToSlide]);

  return (
    <div className="relative -mx-4 -mt-8 overflow-clip sm:-mx-6 lg:-mx-8">
      <nav
        aria-label="Presentation sections"
        className="fixed right-5 top-1/2 z-40 hidden -translate-y-1/2 rounded-md border border-slate-200 bg-white/95 p-2 shadow-lg backdrop-blur lg:block"
      >
        <ol className="space-y-1">
          {slides.map((slide, index) => (
            <li key={slide.id}>
              <button
                type="button"
                aria-label={`Go to ${slide.label}`}
                title={slide.label}
                aria-current={activeSlide === index ? "step" : undefined}
                onClick={() => goToSlide(index)}
                className="flex h-7 w-7 items-center justify-center rounded-full"
              >
                <span
                  aria-hidden="true"
                  className={`block rounded-full transition-all ${activeSlide === index ? "h-2.5 w-2.5 bg-cyan-700" : "h-1.5 w-1.5 bg-slate-300 hover:bg-slate-500"}`}
                />
              </button>
            </li>
          ))}
        </ol>
      </nav>

      <Slide id="opening" index={0} className="flex min-h-[calc(100svh-10rem)] flex-col overflow-hidden bg-slate-950 text-white">
        <video
          className="absolute inset-0 h-full w-full object-cover opacity-55"
          autoPlay
          muted
          loop
          playsInline
          poster={`${imageBase}/presentations/agent-system-layer/images/agent-nebula.png`}
          aria-label="AgentNebula replay of coding-agent file activity"
        >
          <source src="/presentations/agent-system-layer/agent-nebula.mp4" type="video/mp4" />
        </video>
        <div aria-hidden="true" className="absolute inset-0 bg-slate-950/55" />
        <div className="relative z-10 flex flex-1 items-center py-14">
          <div className="max-w-5xl">
            <p className="text-sm font-semibold text-cyan-200">A systems keynote opening</p>
            <h1 className="mt-5 text-4xl font-semibold tracking-normal text-white sm:text-5xl lg:text-6xl">
              The System Layer for AI Agents
            </h1>
            <p className="mt-7 max-w-3xl text-xl leading-8 text-slate-100 sm:text-2xl sm:leading-9">
              AI agents have become long-running operating-system workloads.
            </p>
          </div>
        </div>
      </Slide>

      <Slide id="semantic-gap" index={1} className="bg-white text-slate-950">
        <div className="flex min-h-[calc(100svh-16rem)] flex-col justify-center py-10">
          <h2 className="max-w-4xl text-4xl font-semibold tracking-normal sm:text-5xl">Operating systems execute instructions.</h2>
          <p className="mt-4 max-w-4xl text-4xl font-semibold tracking-normal text-cyan-800 sm:text-5xl">
            Agents execute intentions.
          </p>

          <div className="mt-14 grid gap-7 lg:grid-cols-[1fr_auto_1fr] lg:items-stretch">
            <div className="border-t-4 border-slate-950 py-6">
              <p className="text-sm font-semibold text-slate-500">Today&apos;s OS</p>
              <div className="mt-7 grid grid-cols-2 gap-x-8 gap-y-4 text-2xl font-medium text-slate-900">
                <span>CPU</span><span>Memory</span><span>Processes</span><span>Files</span><span>Sockets</span>
              </div>
            </div>
            <div className="flex min-h-32 items-center justify-center border-y border-dashed border-rose-300 px-8 text-center lg:border-x lg:border-y-0">
              <div>
                <p className="text-5xl font-light text-rose-500">?</p>
                <p className="mt-3 text-sm font-semibold text-rose-700">Semantic gap</p>
              </div>
            </div>
            <div className="border-t-4 border-cyan-700 py-6">
              <p className="text-sm font-semibold text-slate-500">Today&apos;s agent</p>
              <div className="mt-7 grid grid-cols-2 gap-x-8 gap-y-4 text-2xl font-medium text-slate-900">
                <span>Goals</span><span>Intent</span><span>Memory</span><span>Skills</span><span>Tools</span>
              </div>
            </div>
          </div>
        </div>
      </Slide>

      <Slide id="long-running" index={2} className="bg-[#eef6f6] text-slate-950">
        <div className="grid min-h-[calc(100svh-16rem)] gap-12 py-10 lg:grid-cols-[0.8fr_1.2fr] lg:items-center">
          <div>
            <p className="text-sm font-semibold text-cyan-800">Imagine an agent works for</p>
            <p className="mt-3 text-7xl font-semibold text-slate-950 sm:text-8xl">1 week.</p>
            <h2 className="mt-10 text-3xl font-semibold tracking-normal sm:text-4xl">Which actions mattered?</h2>
            <p className="mt-5 max-w-xl text-lg leading-8 text-slate-600">
              A syscall trace can reconstruct execution. It cannot tell you which attempt changed the design, which retry repeated a failure, or why the agent moved its attention to another part of the codebase.
            </p>
          </div>
          <div className="rounded-md border border-slate-800 bg-slate-950 p-6 font-mono text-sm leading-8 text-slate-300 shadow-xl sm:p-8 sm:text-base">
            <p><span className="text-cyan-300">09:12:04</span> read() &nbsp; AGENTS.md</p>
            <p><span className="text-cyan-300">09:14:27</span> execve() &nbsp; cargo test</p>
            <p><span className="text-amber-300">11:38:51</span> write() &nbsp; src/policy.rs</p>
            <p><span className="text-amber-300">14:02:08</span> connect() &nbsp; model API</p>
            <p><span className="text-emerald-300">19:47:33</span> execve() &nbsp; git diff</p>
            <p className="text-slate-500">... processes, files, tools, prompts, retries ...</p>
            <p><span className="text-rose-300">Day 3</span> &nbsp;&nbsp;&nbsp; write() &nbsp; paper/figure-4.pdf</p>
          </div>
        </div>
      </Slide>

      <Slide id="observe" index={3} className="bg-slate-950 text-white">
        <div className="py-9">
          <p className="text-sm font-semibold text-cyan-200">Observe</p>
          <h2 className="mt-4 max-w-4xl text-4xl font-semibold tracking-normal sm:text-5xl">Observe execution semantically.</h2>
          <p className="mt-5 max-w-3xl text-lg leading-8 text-slate-300">
            AgentSight connects prompts, models, and tool decisions to the processes, files, network activity, and resources that carried them out.
          </p>

          <div className="mt-10 grid gap-5 lg:grid-cols-2">
            <Screenshot
              src={`${imageBase}/agentsight/images/top-mode-demo.png`}
              alt="AgentSight top view showing live AI agent sessions"
              caption="Live sessions: model, tokens, health, processes, tools, files, and network activity."
            />
            <figure className="overflow-hidden rounded-md border border-slate-700 bg-black">
              <div className="relative aspect-[16/9] bg-black">
                <video className="h-full w-full object-contain" autoPlay muted loop playsInline poster={`${imageBase}/presentations/agent-system-layer/images/agent-nebula.png`}>
                  <source src="/presentations/agent-system-layer/agent-nebula.mp4" type="video/mp4" />
                </video>
              </div>
              <figcaption className="border-t border-slate-700 px-4 py-3 text-sm leading-6 text-slate-300">
                AgentNebula: replay how attention and file structure changed across a long-running session.
              </figcaption>
            </figure>
            <Screenshot
              src={`${imageBase}/agentsight/flamegraph-example/semantic-flamegraph-top200.svg`}
              alt="AgentSight semantic flamegraph of AI agent activity"
              caption="Semantic flamegraph: connect prompts and tool paths to weighted system effects."
              className="lg:col-span-2"
            />
          </div>
          <div className="mt-7">
            <ProjectLink link={linkByKey.get("agentsight-github")} accent="border-cyan-400/50 bg-cyan-300/10 text-cyan-100 hover:bg-cyan-300/20" />
          </div>
        </div>
      </Slide>

      <Slide id="enforce" index={4} className="bg-white text-slate-950">
        <div className="grid min-h-[calc(100svh-16rem)] gap-10 py-10 lg:grid-cols-[0.8fr_1.2fr] lg:items-center">
          <div>
            <p className="text-sm font-semibold text-rose-700">Enforce</p>
            <h2 className="mt-4 text-4xl font-semibold tracking-normal sm:text-5xl">Visibility is not enough.</h2>
            <p className="mt-5 max-w-xl text-lg leading-8 text-slate-600">
              A useful policy can depend on what happened earlier. If source files changed after the last test, a commit should wait until the new code passes.
            </p>
            <div className="mt-9 space-y-4 font-mono text-base">
              <div className="rounded-md border border-slate-200 bg-slate-50 p-4"><span className="text-slate-500">Agent</span><br />git commit</div>
              <div className="rounded-md border border-rose-200 bg-rose-50 p-4"><span className="text-rose-700">Kernel</span><br />Denied.</div>
              <div className="rounded-md border border-amber-200 bg-amber-50 p-4"><span className="text-amber-800">Reason</span><br />Tests never passed after the latest source change.</div>
            </div>
          </div>
          <Screenshot
            src={`${imageBase}/presentations/agent-system-layer/images/actplane-policy-dsl.png`}
            alt="ActPlane policy DSL compiled into eBPF enforcement"
            caption="A human or parent agent supplies the policy; ActPlane turns it into enforcement at the operating-system boundary."
          />
        </div>
      </Slide>

      <Slide id="policy" index={5} className="bg-[#fff7ed] text-slate-950">
        <div className="py-10">
          <p className="text-sm font-semibold text-orange-800">Policy</p>
          <h2 className="mt-4 max-w-5xl text-4xl font-semibold tracking-normal sm:text-5xl">Policies should become operating-system primitives.</h2>
          <p className="mt-5 max-w-3xl text-lg leading-8 text-slate-700">
            ActPlane carries intent and temporal context into the execution path, then returns a concrete reason when an action violates the rule.
          </p>

          <div className="mt-12 grid gap-8 lg:grid-cols-[0.75fr_1.25fr] lg:items-center">
            <div className="space-y-3">
              {[
                ["Intent", "What outcome is being protected?"],
                ["Context", "What happened earlier in this task?"],
                ["Policy", "What condition must hold now?"],
                ["Execution", "Allow, block, kill, or notify."],
                ["Feedback", "Give the agent a compliant next step."]
              ].map(([name, description], index) => (
                <div key={name} className="grid grid-cols-[2.5rem_1fr] gap-4 border-t border-orange-200 py-3">
                  <span className="font-mono text-sm text-orange-700">{String(index + 1).padStart(2, "0")}</span>
                  <div>
                    <p className="font-semibold text-slate-950">{name}</p>
                    <p className="mt-1 text-sm leading-6 text-slate-600">{description}</p>
                  </div>
                </div>
              ))}
            </div>
            <Screenshot
              src={`${imageBase}/presentations/agent-system-layer/images/actplane-runtime-loop.png`}
              alt="ActPlane runtime architecture with policy, compiler, IFC engine, and feedback"
              caption="The runtime loop keeps policy interpretation above the kernel and enforcement on the path every process must cross."
            />
          </div>
          <div className="mt-7">
            <ProjectLink link={linkByKey.get("actplane-github")} accent="border-orange-300 bg-white text-orange-900 hover:border-orange-500" />
          </div>
        </div>
      </Slide>

      <Slide id="continuity" index={6} className="bg-[#f5f7f6] text-slate-950">
        <div className="grid min-h-[calc(100svh-16rem)] gap-12 py-10 lg:grid-cols-[1fr_1fr] lg:items-center">
          <div>
            <p className="text-sm font-semibold text-emerald-800">Continuity</p>
            <h2 className="mt-4 text-4xl font-semibold tracking-normal sm:text-5xl">Long-running agents deserve continuity.</h2>
            <p className="mt-6 max-w-xl text-lg leading-8 text-slate-600">
              Agent sessions contain decisions, commands, failed attempts, and private working context. Akeep preserves the provider-native files as versioned history so work can be inspected and recovered without turning it into another agent-memory format.
            </p>
            <p className="mt-5 text-sm font-semibold text-amber-800">Akeep is currently pre-alpha.</p>
            <div className="mt-7">
              <ProjectLink link={linkByKey.get("akeep-github")} accent="border-emerald-300 bg-white text-emerald-900 hover:border-emerald-500" />
            </div>
          </div>
          <div className="rounded-md border border-slate-800 bg-slate-950 p-6 shadow-xl sm:p-8">
            <div className="font-mono text-sm leading-8 text-slate-300">
              <p><span className="text-emerald-300">Day 1</span> &nbsp; akeep commit -m &quot;first working path&quot;</p>
              <p><span className="text-emerald-300">Day 2</span> &nbsp; akeep commit -m &quot;after failed migration&quot;</p>
              <p><span className="text-emerald-300">Day 7</span> &nbsp; akeep diff HEAD~1 HEAD</p>
              <p className="mt-5 border-t border-slate-700 pt-5 text-slate-400">Provider changed. Laptop failed. The history remains inspectable.</p>
              <p className="mt-5 text-white">akeep fsck HEAD</p>
              <p className="text-white">akeep checkout HEAD --to /tmp/recovery</p>
              <p className="mt-5 text-emerald-300">Continue.</p>
            </div>
          </div>
        </div>
      </Slide>

      <Slide id="vision" index={7} className="bg-[#102a2b] text-white">
        <div className="flex min-h-[calc(100svh-16rem)] flex-col justify-center py-10">
          <p className="text-sm font-semibold text-emerald-200">Vision</p>
          <h2 className="mt-4 max-w-5xl text-4xl font-semibold tracking-normal sm:text-5xl">The system layer for AI agents.</h2>
          <p className="mt-5 max-w-3xl text-xl leading-8 text-emerald-50">
            We are exploring the missing operating-system abstractions between agent applications and Linux.
          </p>

          <div className="mt-12 border-y border-emerald-200/30">
            <div className="grid gap-4 py-6 md:grid-cols-[13rem_1fr] md:items-center">
              <p className="text-sm font-semibold text-emerald-200">AI applications</p>
              <p className="text-lg text-white">Claude &nbsp; Codex &nbsp; OpenCode &nbsp; Grok &nbsp; Kimi &nbsp; ...</p>
            </div>
            <div className="grid gap-5 border-t border-emerald-200/30 bg-white/5 py-8 md:grid-cols-[13rem_1fr] md:items-start">
              <p className="text-sm font-semibold text-emerald-200">Agent system layer</p>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {[
                  ["Observe", "AgentSight", true],
                  ["Enforce", "ActPlane", true],
                  ["Recover", "Akeep", true],
                  ["Schedule", "Open", false],
                  ["Checkpoint", "Open", false],
                  ["Capability", "Open", false],
                  ["Resource control", "Open", false],
                  ["Isolation", "Open", false]
                ].map(([capability, project, implemented]) => (
                  <div key={capability as string} className={`rounded-md border p-4 ${implemented ? "border-emerald-300/50 bg-emerald-200/10" : "border-white/15 bg-black/10"}`}>
                    <p className="font-semibold text-white">{capability}</p>
                    <p className={`mt-2 text-xs ${implemented ? "text-emerald-200" : "text-slate-400"}`}>{project}</p>
                  </div>
                ))}
              </div>
            </div>
            <div className="grid gap-4 border-t border-emerald-200/30 py-6 md:grid-cols-[13rem_1fr] md:items-center">
              <p className="text-sm font-semibold text-emerald-200">System foundation</p>
              <p className="text-lg text-white">Linux &nbsp; eBPF &nbsp; Containers &nbsp; Filesystems &nbsp; Kernel</p>
            </div>
          </div>

          <p className="mt-10 text-lg text-emerald-50">Open source. Grounded in Linux. eBPF at the execution boundary.</p>
        </div>
      </Slide>
    </div>
  );
}
