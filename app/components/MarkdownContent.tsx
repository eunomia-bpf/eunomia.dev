'use client';

import { useEffect, useMemo, useRef } from 'react';
import DOMPurify from 'dompurify';

type MarkdownContentProps = {
  html: string;
  className?: string;
};

export function MarkdownContent({ html, className }: MarkdownContentProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const codeBlocks = container.querySelectorAll<HTMLElement>('pre > code');

    const cleanups: (() => void)[] = [];

    codeBlocks.forEach((code) => {
      const pre = code.parentElement as HTMLElement;

      // Avoid adding a second button if effect re-runs
      if (pre.querySelector('.copy-code-btn')) return;

      pre.style.position = 'relative';

      const btn = document.createElement('button');
      btn.className = 'copy-code-btn';
      btn.setAttribute('aria-label', 'Copy code to clipboard');
      btn.innerHTML =
        '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg><span>Copy</span>';

      let timeout: ReturnType<typeof setTimeout> | null = null;

      const handleClick = () => {
        const text = code.textContent ?? '';
        navigator.clipboard.writeText(text).then(() => {
          const span = btn.querySelector('span');
          if (span) span.textContent = 'Copied!';
          btn.classList.add('copied');
          if (timeout) clearTimeout(timeout);
          timeout = setTimeout(() => {
            if (span) span.textContent = 'Copy';
            btn.classList.remove('copied');
            timeout = null;
          }, 2000);
        });
      };

      btn.addEventListener('click', handleClick);
      pre.appendChild(btn);

      cleanups.push(() => {
        btn.removeEventListener('click', handleClick);
        if (timeout) clearTimeout(timeout);
        if (pre.contains(btn)) pre.removeChild(btn);
        pre.style.position = '';
      });
    });

    return () => {
      cleanups.forEach((fn) => fn());
    };
  }, [html]);

  const sanitizedHtml = useMemo(() => DOMPurify.sanitize(html), [html]);

  return (
    <div
      ref={containerRef}
      className={["content-copy", className].filter(Boolean).join(" ")}
      dangerouslySetInnerHTML={{ __html: sanitizedHtml }}
    />
  );
}
