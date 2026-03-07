import type { GetServerSideProps } from "next";

import { absoluteUrl } from "../lib/seo";

export const getServerSideProps: GetServerSideProps = async ({ res }) => {
  const body = `User-agent: *
Allow: /

Sitemap: ${absoluteUrl("/sitemap.xml")}
`;
  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.write(body);
  res.end();

  return {
    props: {}
  };
};

export default function Robots() {
  return null;
}
