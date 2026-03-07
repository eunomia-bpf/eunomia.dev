import { createSectionPage } from "../../../lib/page-builders";
import { createSectionArticleRoute } from "../../../lib/route-builders";

const sectionPageRoute = createSectionArticleRoute("zh");
const sectionPage = createSectionPage("zh");

export const getServerSideProps = sectionPageRoute.getServerSideProps;

export default sectionPage;
