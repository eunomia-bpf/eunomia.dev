import type { AppProps } from "next/app";

import { AppErrorBoundary } from "../components/AppErrorBoundary";
import { MermaidHydrator } from "../components/MermaidHydrator";
import "../styles/globals.css";

export default function App({ Component, pageProps, router }: AppProps) {
  return (
    <AppErrorBoundary resetKey={router.asPath}>
      <MermaidHydrator />
      <Component {...pageProps} />
    </AppErrorBoundary>
  );
}
