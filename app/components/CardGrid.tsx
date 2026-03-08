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
    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white">
      {cards.map((card, index) => (
        <a
          key={card.href}
          href={card.href}
          className={`block px-6 py-5 transition hover:bg-slate-50 ${index > 0 ? "border-t border-slate-200" : ""}`}
        >
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <h2 className="text-lg font-semibold tracking-tight text-ink">{card.title}</h2>
              <p className="mt-2 leading-7 text-slate-600">{card.description}</p>
              <span className="mt-4 inline-flex text-sm font-semibold text-slate-900">Open</span>
            </div>
            {card.badge ? (
              <span className="rounded-md bg-slate-100 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-600">
                {card.badge}
              </span>
            ) : null}
          </div>
        </a>
      ))}
    </section>
  );
}
