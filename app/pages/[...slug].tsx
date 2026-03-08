import { createContentPage } from "../lib/page-builders";
import { createContentPageRoute } from "../lib/route-builders";

const contentPageRoute = createContentPageRoute("en");
const contentPage = createContentPage("en");

export const getServerSideProps = contentPageRoute.getServerSideProps;

export default contentPage;
