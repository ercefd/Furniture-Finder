import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

class ImageEditor:
    def __init__(self, device="cpu"):
        self.device = device
        self.pipeline = None
        self.model_id = "timbrooks/instruct-pix2pix"
        
    def load_model(self):
        if self.pipeline is not None:
            return

        print(f"Loading InstructPix2Pix model ({self.model_id}) on {self.device}...")
        try:
            dtype = torch.float16 if self.device == "cuda" else torch.float32 # MPS supports float32 best mostly
            if self.device == "mps":
                dtype = torch.float32
                
            self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.model_id, 
                torch_dtype=dtype, 
                safety_checker=None
            )
            self.pipeline.to(self.device)
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)
            # Optimize for memory if possible
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
                
            print("InstructPix2Pix Loaded Successfully.")
        except Exception as e:
            print(f"Failed to load InstructPix2Pix: {e}")
            self.pipeline = None

    def edit_image(self, image: Image.Image, prompt: str, steps=15, image_guidance_scale=1.2, guidance_scale=8.0):
        if self.pipeline is None:
            self.load_model()
            
        if self.pipeline is None:
            raise RuntimeError("Model execution failed: Model not loaded")

        # Resize for speed/memory if needed (standard SD is 512x512)
        # InstructPix2Pix handles resolutions but 512 is standard
        original_size = image.size
        # Resize maintaining aspect ratio
        w, h = image.size
        # Scale the image so the largest side is 512 pixels
        factor = 512 / max(w, h)
        new_w = int(w * factor)
        new_h = int(h * factor)
        
        # Ensure dimensions are multiples of 8 (requirement for VAE)
        new_w = new_w - (new_w % 8)
        new_h = new_h - (new_h % 8)
        
        input_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        print(f"Editing image: Input {prompt} | Size {new_w}x{new_h}")
        
        result = self.pipeline(
            prompt, 
            image=input_image, 
            num_inference_steps=steps, 
            image_guidance_scale=image_guidance_scale, 
            guidance_scale=guidance_scale
        ).images[0]
        
        # Return to original size
        return result.resize(original_size, Image.Resampling.LANCZOS)

if __name__ == "__main__":
    # Test
    # editor = ImageEditor(device="mps")
    # img = Image.open("test.jpg")
    # res = editor.edit_image(img, "make it look like a oil painting")
    # res.save("edited.jpg")
    pass
