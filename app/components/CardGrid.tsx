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
    <section className="mx-auto grid max-w-6xl gap-6 px-5 md:grid-cols-3">
      {cards.map((card) => (
        <a
          key={card.href}
          href={card.href}
          className="group rounded-[1.75rem] border border-white/70 bg-white/90 p-7 shadow-panel transition duration-300 hover:-translate-y-1 hover:border-azure/30"
        >
          {card.badge ? (
            <span className="inline-flex rounded-full bg-mist px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-azure">
              {card.badge}
            </span>
          ) : null}
          <h2 className="mt-5 text-2xl font-semibold tracking-tight text-ink">{card.title}</h2>
          <p className="mt-4 leading-7 text-slate-600">{card.description}</p>
          <span className="mt-6 inline-flex text-sm font-semibold text-azure transition group-hover:translate-x-1">
            Open
          </span>
        </a>
      ))}
    </section>
  );
}
