from transformers import TrOCRProcessor, TFVisionEncoderDecoderModel
from PIL import Image
import tensorflow as tf

# Load image
img_path = "api/uploads/PLeaseee.heic"
image = Image.open(img_path).convert("RGB")

# Load processor and TensorFlow-based model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = TFVisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Preprocess image and convert to TensorFlow tensor
pixel_values = processor(images=image, return_tensors="tf").pixel_values

# Generate text from the image
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Debugging point
breakpoint()

print(generated_text)
