import { createCollectionPage } from "../../lib/page-builders";
import { createLegacyBlogArticleRoute } from "../../lib/route-builders";
import { loadLegacyBlogIndex, loadLegacyBlogPage } from "../../lib/content";

const legacyBlogPageRoute = createLegacyBlogArticleRoute("en");
const legacyBlogPage = createCollectionPage<
  Awaited<ReturnType<typeof loadLegacyBlogIndex>>,
  NonNullable<Awaited<ReturnType<typeof loadLegacyBlogPage>>>
>("en", "Legacy Blog");

export const getServerSideProps = legacyBlogPageRoute.getServerSideProps;

export default legacyBlogPage;
