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
    <section className="border-t border-slate-200">
      {cards.map((card, index) => (
        <a
          key={card.href}
          href={card.href}
          className={`block py-5 transition hover:bg-slate-50/60 ${index > 0 ? "border-t border-slate-200" : ""}`}
        >
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <h2 className="text-lg font-semibold tracking-tight text-ink">{card.title}</h2>
              <p className="mt-2 max-w-3xl leading-7 text-slate-600">{card.description}</p>
              <span className="mt-3 inline-flex text-sm font-medium text-slate-700">Open</span>
            </div>
            {card.badge ? (
              <span className="whitespace-nowrap text-xs font-medium text-slate-500">
                {card.badge}
              </span>
            ) : null}
          </div>
        </a>
      ))}
    </section>
  );
}
