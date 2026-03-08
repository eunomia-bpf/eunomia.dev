import type { ReactNode } from "react";
import { Component } from "react";

type AppErrorBoundaryProps = {
  children: ReactNode;
  resetKey: string;
};

type AppErrorBoundaryState = {
  hasError: boolean;
};

function localeForPath(pathname: string) {
  return pathname.startsWith("/zh") ? "zh" : "en";
}

function AppErrorFallback({ pathname }: { pathname: string }) {
  const locale = localeForPath(pathname);
  const copy =
    locale === "zh"
      ? {
          badge: "页面暂时不可用",
          title: "这个页面刚刚渲染失败了。",
          body: "文档内容没有丢。你可以刷新当前页面，或者先回到首页继续浏览别的内容。",
          reload: "重新加载",
          home: "返回首页"
        }
      : {
          badge: "Page unavailable",
          title: "This page hit a rendering error.",
          body: "The underlying docs content is still there. Reload this route, or jump back to the homepage and keep browsing.",
          reload: "Reload page",
          home: "Back to home"
        };

  return (
    <main className="min-h-screen bg-paper px-5 py-16 text-ink">
      <section className="mx-auto max-w-3xl rounded-xl border border-slate-200 bg-white p-8 md:p-12">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">{copy.badge}</p>
        <h1 className="mt-4 text-4xl font-semibold leading-tight tracking-tight md:text-5xl">{copy.title}</h1>
        <p className="mt-5 max-w-2xl text-base leading-8 text-slate-600 md:text-lg">{copy.body}</p>
        <div className="mt-8 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="inline-flex rounded-lg bg-ink px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800"
          >
            {copy.reload}
          </button>
          <a
            href={locale === "zh" ? "/zh/" : "/"}
            className="inline-flex rounded-lg border border-slate-200 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 hover:text-ink"
          >
            {copy.home}
          </a>
        </div>
      </section>
    </main>
  );
}

export class AppErrorBoundary extends Component<AppErrorBoundaryProps, AppErrorBoundaryState> {
  state: AppErrorBoundaryState = {
    hasError: false
  };

  static getDerivedStateFromError() {
    return {
      hasError: true
    };
  }

  componentDidCatch(error: unknown, info: unknown) {
    console.error("AppErrorBoundary caught a rendering error.", error, info);
  }

  componentDidUpdate(previousProps: AppErrorBoundaryProps) {
    if (this.state.hasError && previousProps.resetKey !== this.props.resetKey) {
      this.setState({ hasError: false });
    }
  }

  render() {
    if (this.state.hasError) {
      return <AppErrorFallback pathname={this.props.resetKey} />;
    }

    return this.props.children;
  }
}
