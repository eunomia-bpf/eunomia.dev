import { createCollectionPage } from "../../lib/page-builders";
import { createTutorialPageRoute } from "../../lib/route-builders";
import { loadTutorialIndex, loadTutorialPage } from "../../lib/content";

const tutorialsPageRoute = createTutorialPageRoute("en");
const tutorialsPage = createCollectionPage<
  Awaited<ReturnType<typeof loadTutorialIndex>>,
  NonNullable<Awaited<ReturnType<typeof loadTutorialPage>>>
>("en", "Tutorials");

export const getStaticPaths = tutorialsPageRoute.getStaticPaths;
export const getStaticProps = tutorialsPageRoute.getStaticProps;

export default tutorialsPage;
