import { createContentPage } from "../../lib/page-builders";
import { createContentPageRoute } from "../../lib/route-builders";

const contentPageRoute = createContentPageRoute("zh");
const contentPage = createContentPage("zh");

export const getServerSideProps = contentPageRoute.getServerSideProps;

export default contentPage;
