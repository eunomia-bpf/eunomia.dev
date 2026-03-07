import { siteConfig } from "../lib/site-data";

export function SiteFooter() {
  return (
    <footer className="mt-16 border-t border-slate-200 bg-white/80">
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-5 py-10 text-sm text-slate-600 md:flex-row md:items-center md:justify-between">
        <p>Custom frontend migration slice for eunomia.dev.</p>
        <div className="flex gap-4">
          <a href={siteConfig.repoUrl} className="transition hover:text-azure">
            GitHub
          </a>
          <a href="https://github.com/orgs/eunomia-bpf/discussions" className="transition hover:text-azure">
            Discussion
          </a>
        </div>
      </div>
    </footer>
  );
}
