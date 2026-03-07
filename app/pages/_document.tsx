import Document, {
  Head,
  Html,
  Main,
  NextScript,
  type DocumentContext,
  type DocumentInitialProps
} from "next/document";

type Props = DocumentInitialProps & {
  localeLang: "en" | "zh";
};

export default class MyDocument extends Document<Props> {
  static async getInitialProps(ctx: DocumentContext): Promise<Props> {
    const initialProps = await Document.getInitialProps(ctx);
    const requestPath = ctx.req?.url ?? ctx.pathname ?? "/";
    const localeLang = requestPath.startsWith("/zh") ? "zh" : "en";
    return {
      ...initialProps,
      localeLang
    };
  }

  render() {
    return (
      <Html lang={this.props.localeLang}>
        <Head />
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    );
  }
}
