/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  distDir: process.env.NEXT_DIST_DIR ?? ".next",
  output: "export",
  experimental: {
    largePageDataBytes: 256 * 1024
  },
  trailingSlash: true
};

export default nextConfig;
