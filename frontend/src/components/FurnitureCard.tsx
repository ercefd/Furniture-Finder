interface FurnitureCardProps {
  item: {
    id: string | number;
    name: string;
    image_url: string;
    score: number;
  };
}

export default function FurnitureCard({ item }: FurnitureCardProps) {
  return (
    <div className="furniture-card group">
      <div className="relative overflow-hidden">
        <img
          src={item.image_url}
          alt={item.name}
          className="w-full h-48 object-cover transition-transform duration-500 group-hover:scale-110"
          onError={(e) => {
            (e.target as HTMLImageElement).src = 'https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=400&h=300&fit=crop';
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-background/90 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        
        {/* Score badge */}
        <div className="absolute top-3 right-3 px-3 py-1 rounded-full bg-primary/20 backdrop-blur-md border border-primary/30">
          <span className="text-xs font-medium text-primary">
            {(item.score * 100).toFixed(1)}% match
          </span>
        </div>
      </div>
      
      <div className="p-4 space-y-2">
        <h3 className="font-semibold text-foreground line-clamp-1 group-hover:text-primary transition-colors">
          {item.name}
        </h3>
        <div className="flex items-center gap-2">
          <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-primary to-accent rounded-full transition-all duration-500"
              style={{ width: `${item.score * 100}%` }}
            />
          </div>
          <span className="text-xs text-muted-foreground">
            {item.score.toFixed(3)}
          </span>
        </div>
      </div>
    </div>
  );
}
