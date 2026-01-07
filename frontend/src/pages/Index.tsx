import { useState, Suspense } from 'react';
import axios from 'axios';
import { Search, Loader2, Zap, Cpu, Eye } from 'lucide-react';
import { Button } from '@/components/ui/button';
import UploadZone from '@/components/UploadZone';
import CaptionDisplay from '@/components/CaptionDisplay';
import ResultsGrid from '@/components/ResultsGrid';
import Robot3D from '@/components/Robot3D';
import { useToast } from '@/hooks/use-toast';

interface SearchResult {
  id: string | number;
  name: string;
  image_url: string;
  score: number;
}

const Index = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [caption, setCaption] = useState('');
  const [loading, setLoading] = useState(false);
  const [textQuery, setTextQuery] = useState('');
  const { toast } = useToast();

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResults([]);
    setCaption('');
  };

  const handleTextSearch = async () => {
    if (!textQuery.trim()) return;

    setLoading(true);
    setResults([]); // Clear previous results
    
    try {
      const response = await axios.get(`http://localhost:8000/search_text?q=${encodeURIComponent(textQuery)}`);
      setResults(response.data.results || []);
      
      toast({
         title: 'Search Complete',
         description: `Found ${response.data.results?.length || 0} items for "${textQuery}"`
      });
    } catch (error) {
       console.error("Text search error:", error);
       toast({
        title: 'Search Failed',
        description: 'Could not perform text search.',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const searchReq = axios.post('http://localhost:8000/search', formData);
      const captionReq = axios.post('http://localhost:8000/caption', formData);

      const [searchRes, captionRes] = await Promise.all([searchReq, captionReq]);

      setResults(searchRes.data.results || []);
      setCaption(captionRes.data.caption || 'No caption generated.');

      toast({
        title: 'Analysis Complete',
        description: `Found ${searchRes.data.results?.length || 0} similar items`,
      });
    } catch (error) {
      console.error('Error searching:', error);
      toast({
        title: 'Connection Error',
        description: 'Failed to connect to backend. Ensure the server is running on localhost:8000',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };


  // Image Editing Logic Removed

  return (
    <div className="min-h-screen">
      {/* Background effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl animate-glow-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-accent/5 rounded-full blur-3xl animate-glow-pulse" style={{ animationDelay: '1s' }} />
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-8">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary/50 border border-border/50 mb-6">
            <Zap className="w-4 h-4 text-primary" />
            <span className="text-sm text-muted-foreground">AI-Powered Visual Search</span>
          </div>
          <h1 className="text-4xl md:text-6xl font-bold mb-4">
            <span className="text-foreground">Furniture</span>{' '}
            <span className="glow-text">Prompter</span>
          </h1>

          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Upload any furniture image and let our AI analyze it, generate descriptions,
            and find similar pieces from our database.
          </p>
        </header>

        {/* Robot and Features */}
        <div className="grid lg:grid-cols-2 gap-8 mb-12 items-center">
          <div className="order-2 lg:order-1">
            <div className="space-y-6">
              <div className="glass-card p-6 flex items-start gap-4">
                <div className="p-3 rounded-xl bg-primary/10 shrink-0">
                  <Eye className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground mb-1">Visual Analysis</h3>
                  <p className="text-sm text-muted-foreground">
                    Advanced image recognition identifies furniture style, materials, and design elements.
                  </p>
                </div>
              </div>

              <div className="glass-card p-6 flex items-start gap-4">
                <div className="p-3 rounded-xl bg-accent/10 shrink-0">
                  <Cpu className="w-6 h-6 text-accent" />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground mb-1">AI Description</h3>
                  <p className="text-sm text-muted-foreground">
                    Generate detailed natural language descriptions of your furniture pieces.
                  </p>
                </div>
              </div>

              <div className="glass-card p-6 flex items-start gap-4">
                <div className="p-3 rounded-xl bg-primary/10 shrink-0">
                  <Search className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground mb-1">Similarity Search</h3>
                  <p className="text-sm text-muted-foreground">
                    Find visually similar furniture from our extensive database with precision scoring.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="order-1 lg:order-2">
            <Suspense fallback={
              <div className="w-full h-[400px] flex items-center justify-center">
                <Loader2 className="w-8 h-8 text-primary animate-spin" />
              </div>
            }>
              <Robot3D />
            </Suspense>
          </div>
        </div>

        {/* Search Options */}
        <section className="max-w-3xl mx-auto mb-12">
           <div className="glass-card p-6 mb-8 animate-fade-in">
              <div className="flex items-center gap-3 mb-4">
                 <Search className="w-5 h-5 text-primary" />
                 <h3 className="font-semibold text-lg">Text Search</h3>
              </div>
              <div className="flex gap-3">
                 <input
                   type="text"
                   placeholder="Describe what you are looking for (e.g., 'red velvet sofa', 'modern wooden table')..."
                   className="flex-1 bg-background/50 border border-border/50 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary/50"
                   value={textQuery}
                   onChange={(e) => setTextQuery(e.target.value)}
                   onKeyDown={(e) => e.key === 'Enter' && handleTextSearch()}
                 />
                 <Button onClick={handleTextSearch} disabled={loading || !textQuery}>
                    {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Search"}
                 </Button>
              </div>
           </div>

           <div className="flex items-center gap-4 mb-6">
              <div className="h-px bg-border flex-1" />
              <span className="text-muted-foreground text-sm font-medium">OR UPLOAD IMAGE</span>
              <div className="h-px bg-border flex-1" />
           </div>

          <UploadZone preview={preview} onFileSelect={handleFileSelect} />

          {selectedFile && (
            <div className="text-center mt-6 animate-fade-in">
              <Button
                variant="hero"
                onClick={handleSearch}
                disabled={loading}
                className="min-w-[200px]"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Search className="w-5 h-5" />
                    Analyze & Search
                  </>
                )}
              </Button>
            </div>
          )}


          <CaptionDisplay caption={caption} />
        </section>

        {/* Results */}
        <ResultsGrid results={results} />

        {/* Footer */}
        <footer className="mt-20 text-center pb-8">
          <div className="h-px w-full max-w-md mx-auto bg-gradient-to-r from-transparent via-border to-transparent mb-8" />
          <p className="text-sm text-muted-foreground">
            Powered by AI visual recognition technology
          </p>
        </footer>
      </div>
    </div>
  );
};

export default Index;
