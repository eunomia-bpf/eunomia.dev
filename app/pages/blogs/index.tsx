import { createCollectionPage } from "../../lib/page-builders";
import { createLegacyBlogPageRoute } from "../../lib/route-builders";
import { loadLegacyBlogIndex, loadLegacyBlogPage } from "../../lib/content";

const legacyBlogPageRoute = createLegacyBlogPageRoute("en");
const legacyBlogPage = createCollectionPage<
  Awaited<ReturnType<typeof loadLegacyBlogIndex>>,
  NonNullable<Awaited<ReturnType<typeof loadLegacyBlogPage>>>
>("en", "Legacy Blog");

export const getStaticProps = legacyBlogPageRoute.getStaticProps;

export default legacyBlogPage;
