/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  distDir: process.env.NEXT_DIST_DIR ?? ".next",
  experimental: {
    largePageDataBytes: 256 * 1024
  },
  trailingSlash: true
};

export default nextConfig;
