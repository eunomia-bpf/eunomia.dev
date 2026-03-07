import { createCollectionPage } from "../../../lib/page-builders";
import { createBlogPageRoute } from "../../../lib/route-builders";
import { loadBlogIndex, loadBlogPage } from "../../../lib/content";

const blogPageRoute = createBlogPageRoute("zh");
const blogPage = createCollectionPage<
  Awaited<ReturnType<typeof loadBlogIndex>>,
  NonNullable<Awaited<ReturnType<typeof loadBlogPage>>>
>("zh", "博客");

export const getStaticProps = blogPageRoute.getStaticProps;

export default blogPage;
