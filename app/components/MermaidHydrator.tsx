"use client";

import { useEffect } from "react";
import { useRouter } from "next/router";

let mermaidLoader: Promise<typeof import("mermaid").default> | null = null;
let mermaidInitialized = false;

async function loadMermaid() {
  mermaidLoader ??= import("mermaid").then((module) => module.default);
  return mermaidLoader;
}

function makeDiagramId(pathname: string, index: number) {
  return `mermaid-${pathname.replace(/[^a-zA-Z0-9_-]+/g, "-")}-${index}`;
}

export function MermaidHydrator() {
  const router = useRouter();

  useEffect(() => {
    let cancelled = false;

    async function hydrateMermaid() {
      const targets = Array.from(
        document.querySelectorAll<HTMLElement>("pre.mermaid-diagram:not([data-mermaid-rendered])")
      );
      if (!targets.length) {
        return;
      }

      const mermaid = await loadMermaid();
      if (!mermaidInitialized) {
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: "neutral",
          fontFamily: "ui-sans-serif, system-ui, sans-serif"
        });
        mermaidInitialized = true;
      }

      for (const [index, target] of targets.entries()) {
        const source = target.textContent?.trim();
        if (!source) {
          continue;
        }

        target.setAttribute("data-mermaid-rendered", "pending");

        try {
          const { svg, bindFunctions } = await mermaid.render(makeDiagramId(router.asPath, index), source);
          if (cancelled) {
            return;
          }

          const wrapper = document.createElement("div");
          wrapper.className = "mermaid-rendered";
          wrapper.setAttribute("data-mermaid-rendered", "true");
          wrapper.innerHTML = svg;
          target.replaceWith(wrapper);
          bindFunctions?.(wrapper);
        } catch (error) {
          target.setAttribute("data-mermaid-rendered", "failed");
          console.error("Mermaid render failed.", error);
        }
      }
    }

    void hydrateMermaid();
    return () => {
      cancelled = true;
    };
  }, [router.asPath]);

  return null;
}
