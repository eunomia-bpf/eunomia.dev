import { createCollectionPage } from "../../lib/page-builders";
import { createBlogPageRoute } from "../../lib/route-builders";
import { loadBlogIndex, loadBlogPage } from "../../lib/content";

const blogPageRoute = createBlogPageRoute("en");
const blogPage = createCollectionPage<
  Awaited<ReturnType<typeof loadBlogIndex>>,
  NonNullable<Awaited<ReturnType<typeof loadBlogPage>>>
>("en", "Blog");

export const getStaticPaths = blogPageRoute.getStaticPaths;
export const getStaticProps = blogPageRoute.getStaticProps;

export default blogPage;
