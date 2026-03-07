type MarkdownContentProps = {
  html: string;
};

export function MarkdownContent({ html }: MarkdownContentProps) {
  return <div className="content-copy" dangerouslySetInnerHTML={{ __html: html }} />;
}
