/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  trailingSlash: true,
  async redirects() {
    return [
      {
        source: "/blogs/:path*",
        destination: "/blog/:path*",
        permanent: true
      }
    ];
  }
};

export default nextConfig;
