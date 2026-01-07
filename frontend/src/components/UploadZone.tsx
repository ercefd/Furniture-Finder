import { useRef, useState } from 'react';
import { Upload, Image as ImageIcon } from 'lucide-react';

interface UploadZoneProps {
  preview: string | null;
  onFileSelect: (file: File) => void;
}

export default function UploadZone({ preview, onFileSelect }: UploadZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  return (
    <div
      className={`upload-zone p-8 md:p-12 min-h-[280px] flex items-center justify-center cursor-pointer transition-all duration-300 ${
        isDragOver ? 'dragover border-primary shadow-glow-lg' : ''
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <input
        type="file"
        hidden
        ref={fileInputRef}
        onChange={handleFileSelect}
        accept="image/*"
      />
      
      {preview ? (
        <div className="relative group">
          <img
            src={preview}
            alt="Upload Preview"
            className="max-h-[300px] rounded-xl object-contain transition-transform duration-300 group-hover:scale-[1.02]"
          />
          <div className="absolute inset-0 bg-background/50 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl flex items-center justify-center">
            <p className="text-foreground font-medium">Click to change image</p>
          </div>
        </div>
      ) : (
        <div className="text-center space-y-4">
          <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-6">
            <Upload className="w-8 h-8 text-primary" />
          </div>
          <div className="space-y-2">
            <p className="text-lg font-medium text-foreground">
              Drop your furniture image here
            </p>
            <p className="text-muted-foreground">
              or click to browse your files
            </p>
          </div>
          <div className="flex items-center justify-center gap-2 pt-4">
            <ImageIcon className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">
              Supports JPG, PNG, WebP
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
