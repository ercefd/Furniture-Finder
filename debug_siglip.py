from transformers import SiglipProcessor, SiglipModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Loading...")
try:
    processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
    print("Processor loaded")
    model = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
    print("Model loaded")
except Exception as e:
    print(e)
