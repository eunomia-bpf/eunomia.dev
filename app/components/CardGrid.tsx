type Card = {
  title: string;
  description: string;
  href: string;
  badge?: string;
};

type CardGridProps = {
  cards: Card[];
};

export function CardGrid({ cards }: CardGridProps) {
  return (
    <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      {cards.map((card) => (
        <a
          key={card.href}
          href={card.href}
          className="group rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition hover:border-slate-300"
        >
          {card.badge ? (
            <span className="inline-flex rounded-full bg-slate-100 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-600">
              {card.badge}
            </span>
          ) : null}
          <h2 className="mt-4 text-xl font-semibold tracking-tight text-ink">{card.title}</h2>
          <p className="mt-3 leading-7 text-slate-600">{card.description}</p>
          <span className="mt-5 inline-flex text-sm font-semibold text-slate-900">
            Open
          </span>
        </a>
      ))}
    </section>
  );
}
