import { createCollectionPage } from "../../lib/page-builders";
import { createTutorialArticleRoute } from "../../lib/route-builders";
import { loadTutorialIndex, loadTutorialPage } from "../../lib/content";

const tutorialsPageRoute = createTutorialArticleRoute("en");
const tutorialsPage = createCollectionPage<
  Awaited<ReturnType<typeof loadTutorialIndex>>,
  NonNullable<Awaited<ReturnType<typeof loadTutorialPage>>>
>("en", "Tutorials");

export const getServerSideProps = tutorialsPageRoute.getServerSideProps;

export default tutorialsPage;
