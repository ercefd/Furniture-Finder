from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class PromptGenerator:
    def __init__(self, model_name="Salesforce/blip-image-captioning-large", device="cpu"):
        self.device = device
        self.model = None
        try:
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        except Exception as e:
            print(f"Failed to load BLIP model: {e}")
            self.model = None
    
    def generate_prompt(self, image: Image.Image):
        if self.model is None:
            return "Captioning unavailable (Model failed to load)"
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # Generate with parameters for more detail
        out = self.model.generate(
            **inputs, 
            max_length=100, 
            min_length=20, 
            num_beams=5, 
            repetition_penalty=1.2
        )
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption


if __name__ == "__main__":
    # Test
    # gen = PromptGenerator()
    # print(gen.generate_prompt(Image.open("test.jpg")))
    pass
