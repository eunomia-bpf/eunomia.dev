import type { GetServerSideProps } from "next";

import { renderFeed } from "../../lib/content/feed";

export const getServerSideProps: GetServerSideProps = async ({ res }) => {
  res.setHeader("Content-Type", "application/rss+xml; charset=utf-8");
  res.write(renderFeed("zh"));
  res.end();

  return {
    props: {}
  };
};

export default function Feed() {
  return null;
}
