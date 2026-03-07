import { createCollectionPage } from "../../../lib/page-builders";
import { createBlogArticleRoute } from "../../../lib/route-builders";
import { loadBlogIndex, loadBlogPage } from "../../../lib/content";

const blogPageRoute = createBlogArticleRoute("zh");
const blogPage = createCollectionPage<
  Awaited<ReturnType<typeof loadBlogIndex>>,
  NonNullable<Awaited<ReturnType<typeof loadBlogPage>>>
>("zh", "博客");

export const getServerSideProps = blogPageRoute.getServerSideProps;

export default blogPage;
