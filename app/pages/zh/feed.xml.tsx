import { createFeedPage } from "../../lib/page-builders";
import { createFeedPageRoute } from "../../lib/route-builders";

const feedPageRoute = createFeedPageRoute("zh");
const feedPage = createFeedPage();

export const getServerSideProps = feedPageRoute.getServerSideProps;

export default feedPage;
