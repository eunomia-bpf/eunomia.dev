import Link from "next/link";

export default function ServerErrorPage() {
  return (
    <main className="min-h-screen bg-paper px-5 py-16 text-ink">
      <section className="mx-auto max-w-3xl rounded-[2.5rem] border border-white/70 bg-white/90 p-8 shadow-panel md:p-12">
        <span className="inline-flex rounded-full bg-ink px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] text-white">
          Internal error
        </span>
        <h1 className="mt-6 font-serif text-4xl leading-tight tracking-tight md:text-5xl">
          The server could not finish rendering this page.
        </h1>
        <p className="mt-5 max-w-2xl text-base leading-8 text-slate-600 md:text-lg">
          Try reloading the route. If the error persists, return to the homepage and keep browsing another section while
          this page is fixed.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link
            href="/"
            className="inline-flex rounded-full bg-ink px-5 py-3 text-sm font-semibold text-white transition hover:bg-azure"
          >
            Back to home
          </Link>
          <Link
            href="/zh/"
            className="inline-flex rounded-full border border-slate-200 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:border-azure hover:text-azure"
          >
            前往中文首页
          </Link>
        </div>
      </section>
    </main>
  );
}
