import { createCollectionPage } from "../../../lib/page-builders";
import { createLegacyBlogPageRoute } from "../../../lib/route-builders";
import { loadLegacyBlogIndex, loadLegacyBlogPage } from "../../../lib/content";

const legacyBlogPageRoute = createLegacyBlogPageRoute("zh");
const legacyBlogPage = createCollectionPage<
  Awaited<ReturnType<typeof loadLegacyBlogIndex>>,
  NonNullable<Awaited<ReturnType<typeof loadLegacyBlogPage>>>
>("zh", "旧博客");

export const getStaticPaths = legacyBlogPageRoute.getStaticPaths;
export const getStaticProps = legacyBlogPageRoute.getStaticProps;

export default legacyBlogPage;
