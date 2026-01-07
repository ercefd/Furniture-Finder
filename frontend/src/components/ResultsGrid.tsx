import FurnitureCard from './FurnitureCard';

interface Result {
  id: string | number;
  name: string;
  image_url: string;
  score: number;
}

interface ResultsGridProps {
  results: Result[];
}

export default function ResultsGrid({ results }: ResultsGridProps) {
  if (results.length === 0) return null;

  return (
    <section className="mt-12 animate-fade-in">
      <div className="flex items-center gap-3 mb-6">
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-border to-transparent" />
        <h2 className="text-xl font-semibold text-foreground">
          Similar Furniture Found
        </h2>
        <div className="h-px flex-1 bg-gradient-to-l from-transparent via-border to-transparent" />
      </div>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {results.map((item, index) => (
          <div
            key={item.id}
            className="animate-scale-in"
            style={{ animationDelay: `${index * 100}ms` }}
          >
            <FurnitureCard item={item} />
          </div>
        ))}
      </div>
    </section>
  );
}
