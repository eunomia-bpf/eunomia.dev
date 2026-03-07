import { createSectionPage } from "../../../lib/page-builders";
import { createSectionPageRoute } from "../../../lib/route-builders";

const sectionPageRoute = createSectionPageRoute("zh");
const sectionPage = createSectionPage("zh");

export const getStaticPaths = sectionPageRoute.getStaticPaths;
export const getStaticProps = sectionPageRoute.getStaticProps;

export default sectionPage;
