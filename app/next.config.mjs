import { PHASE_DEVELOPMENT_SERVER } from "next/constants.js";

/** @type {import('next').NextConfig} */
export default function nextConfig(phase) {
  const isDev = phase === PHASE_DEVELOPMENT_SERVER;
  const distDir = process.env.NEXT_DIST_DIR ?? (isDev ? ".next-dev" : ".next");

  return {
    reactStrictMode: true,
    distDir,
    // Keep local development on the standard Next runtime; static export remains
    // enforced for production builds and deployment verification.
    output: isDev ? undefined : "export",
    experimental: {
      largePageDataBytes: 256 * 1024
    },
    trailingSlash: true
  };
}
