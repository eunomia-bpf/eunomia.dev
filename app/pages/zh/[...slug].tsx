import { createContentPage } from "../../lib/page-builders";
import { createContentPageRoute } from "../../lib/route-builders";

const contentPageRoute = createContentPageRoute("zh");
const contentPage = createContentPage("zh");

export const getStaticPaths = contentPageRoute.getStaticPaths;
export const getStaticProps = contentPageRoute.getStaticProps;

export default contentPage;
