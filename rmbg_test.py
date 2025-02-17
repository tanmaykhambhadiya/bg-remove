import torch
import torch.nn.functional as F
import numpy as np
import requests
from PIL import Image
from torchvision.transforms.functional import normalize
from transformers import AutoModelForImageSegmentation
from urllib.parse import urlparse
import os
from io import BytesIO
from skimage import io

# ✅ Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True).to(device)

# ✅ Function to check if the image is local or a URL
def load_image(image_path):
    if os.path.exists(image_path):  # Local file
        return Image.open(image_path).convert("RGB")
    else:
        parsed_url = urlparse(image_path)
        if parsed_url.scheme in ["http", "https"]:  # URL
            response = requests.get(image_path)
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            raise ValueError(f"Invalid image path: {image_path}. Provide a valid file path or URL.")

# ✅ Preprocessing function
def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]  # Convert grayscale to RGB
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)  # Convert to tensor
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor, 255.0)  # Normalize to [0,1]
    image = normalize(image, [0.5], [1.0])  # Normalize for model input
    return image

# ✅ Postprocessing function
def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
    result = (result - result.min()) / (result.max() - result.min())  # Normalize mask
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    return np.squeeze(im_array)  # Convert to NumPy array

# ✅ Set image path (local file or URL)
image_path = "24.jpg"  # Replace with a valid local file or image URL

# ✅ Load and process the image
orig_image = load_image(image_path)
orig_im = np.array(orig_image)
orig_im_size = orig_im.shape[0:2]
model_input_size = [1024, 1024]  # Resize input for model

image = preprocess_image(orig_im, model_input_size).to(device)

# ✅ Run inference
with torch.no_grad():
    result = model(image)

# ✅ Process and apply mask
result_image = postprocess_image(result[0][0], orig_im_size)
mask_image = Image.fromarray(result_image)  # Convert mask to PIL image

# ✅ Apply the mask to the original image
no_bg_image = orig_image.copy()
no_bg_image.putalpha(mask_image)  # Add alpha channel (transparency)

# ✅ Save or show result
no_bg_image.save("output_no_bg.png")  # Save the output image
no_bg_image.show()  # Display result

print("✅ Background removed successfully! Saved as 'output_no_bg.png'.")
