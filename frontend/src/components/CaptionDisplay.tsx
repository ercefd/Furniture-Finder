import { Sparkles } from 'lucide-react';

interface CaptionDisplayProps {
  caption: string;
}

export default function CaptionDisplay({ caption }: CaptionDisplayProps) {
  if (!caption) return null;

  return (
    <div className="glass-card p-6 mt-8 animate-fade-in">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-primary/10">
          <Sparkles className="w-5 h-5 text-primary" />
        </div>
        <h3 className="font-semibold glow-text">AI Generated Description</h3>
      </div>
      <p className="text-muted-foreground leading-relaxed">
        {caption}
      </p>
    </div>
  );
}
