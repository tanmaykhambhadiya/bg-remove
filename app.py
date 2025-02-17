# import torch
# import torch.nn.functional as F
# import numpy as np
# import requests
# from PIL import Image
# from torchvision.transforms.functional import normalize
# from transformers import AutoModelForImageSegmentation
# from urllib.parse import urlparse
# import os
# from io import BytesIO
# from flask import Flask, request, jsonify, send_file

# # ✅ Initialize Flask app
# app = Flask(__name__)

# # ✅ Load the model
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True).to(device)

# # ✅ Function to check if the image is local or a URL
# def load_image(image_path):
#     parsed_url = urlparse(image_path)
#     if parsed_url.scheme in ["http", "https"]:  # URL
#         response = requests.get(image_path)
#         return Image.open(BytesIO(response.content)).convert("RGB")
#     else:  # Local file
#         return Image.open(image_path).convert("RGB")

# # ✅ Preprocessing function
# def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
#     if len(im.shape) < 3:
#         im = im[:, :, np.newaxis]  # Convert grayscale to RGB
#     im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)  # Convert to tensor
#     im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
#     image = torch.divide(im_tensor, 255.0)  # Normalize to [0,1]
#     image = normalize(image, [0.5], [1.0])  # Normalize for model input
#     return image

# # ✅ Postprocessing function
# def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
#     result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
#     result = (result - result.min()) / (result.max() - result.min())  # Normalize mask
#     im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
#     return np.squeeze(im_array)  # Convert to NumPy array

# # ✅ API Endpoint: Background Removal
# @app.route("/remove-bg", methods=["POST"])
# def remove_bg():
#     try:
#         # ✅ Check if the request contains an image file
#         if "file" in request.files:
#             file = request.files["file"]
#             image = Image.open(file).convert("RGB")
#         elif "image_url" in request.json:
#             image_url = request.json["image_url"]
#             image = load_image(image_url)
#         else:
#             return jsonify({"error": "No image provided. Use 'file' or 'image_url'."}), 400
        
#         orig_im = np.array(image)
#         orig_im_size = orig_im.shape[0:2]
#         model_input_size = [1024, 1024]

#         # ✅ Preprocess image
#         image_tensor = preprocess_image(orig_im, model_input_size).to(device)

#         # ✅ Run inference
#         with torch.no_grad():
#             result = model(image_tensor)

#         # ✅ Process and apply mask
#         result_image = postprocess_image(result[0][0], orig_im_size)
#         mask_image = Image.fromarray(result_image)

#         # ✅ Apply the mask to the original image
#         no_bg_image = image.copy()
#         no_bg_image.putalpha(mask_image)  # Add transparency

#         # ✅ Save and return response
#         output_path = "output_no_bg.png"
#         no_bg_image.save(output_path)
        
#         return send_file(output_path, mimetype="image/png")

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # ✅ Run the Flask app
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)



# import torch
# import torch.nn.functional as F
# import numpy as np
# import requests
# from PIL import Image
# from torchvision.transforms.functional import normalize
# from transformers import AutoModelForImageSegmentation
# from flask import Flask, request, jsonify, send_file
# from io import BytesIO
# import os
# from urllib.parse import urlparse

# # ✅ Initialize Flask App
# app = Flask(__name__)

# # ✅ Define a Secure API Key
# API_KEY = "C5v9#zF!pQ4&kX8@Jd2M$yN6*WbL3R7%ThG0VZ1^qKxPYmT"  # Change this to a strong key

# # ✅ Load the Model
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True).to(device)

# # ✅ Function to Load Image (Local or URL)
# def load_image(image_path):
#     if os.path.exists(image_path):  # Local file
#         return Image.open(image_path).convert("RGB")
#     parsed_url = urlparse(image_path)
#     if parsed_url.scheme in ["http", "https"]:  # URL
#         response = requests.get(image_path)
#         return Image.open(BytesIO(response.content)).convert("RGB")
#     raise ValueError(f"Invalid image path: {image_path}. Provide a valid file path or URL.")

# # ✅ Preprocessing Function
# def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
#     if len(im.shape) < 3:
#         im = im[:, :, np.newaxis]  # Convert grayscale to RGB
#     im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)  # Convert to tensor
#     im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
#     image = torch.divide(im_tensor, 255.0)  # Normalize to [0,1]
#     image = normalize(image, [0.5], [1.0])  # Normalize for model input
#     return image.to(device)

# # ✅ Postprocessing Function
# def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
#     result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
#     result = (result - result.min()) / (result.max() - result.min())  # Normalize mask
#     im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
#     return np.squeeze(im_array)  # Convert to NumPy array

# # ✅ Remove Background API (Requires API Key)
# @app.route("/remove-bg", methods=["POST"])
# def remove_bg():
#     try:
#         # ✅ Check for API Key in Headers
#         api_key = request.headers.get("Authorization")
#         if not api_key or api_key != f"Bearer {API_KEY}":
#             return jsonify({"error": "Unauthorized. Invalid API key."}), 403

#         # ✅ Process Image File Upload or URL
#         if "file" in request.files:  # Image from form-data (File Upload)
#             image = Image.open(request.files["file"]).convert("RGB")
#         else:  # Image from JSON (URL)
#             data = request.get_json()
#             image_url = data.get("image_url")
#             image = load_image(image_url)

#         orig_im = np.array(image)
#         orig_im_size = orig_im.shape[0:2]
#         model_input_size = [1024, 1024]

#         # ✅ Process Image
#         image_tensor = preprocess_image(orig_im, model_input_size)
        
#         # ✅ Run Model
#         with torch.no_grad():
#             result = model(image_tensor)

#         # ✅ Generate Transparent Image
#         result_image = postprocess_image(result[0][0], orig_im_size)
#         mask_image = Image.fromarray(result_image)

#         no_bg_image = image.copy()
#         no_bg_image.putalpha(mask_image)  # Apply alpha transparency

#         # ✅ Save and Return the Result
#         img_io = BytesIO()
#         no_bg_image.save(img_io, format="PNG")
#         img_io.seek(0)

#         return send_file(img_io, mimetype="image/png")

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # ✅ Run Flask App
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)



import torch
import torch.nn.functional as F
import numpy as np
import requests
from PIL import Image
from torchvision.transforms.functional import normalize
from transformers import AutoModelForImageSegmentation
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import os
from urllib.parse import urlparse


app = Flask(__name__)
 
API_KEY = "C5v9#zF!pQ4&kX8@Jd2M$yN6*WbL3R7%ThG0VZ1^qKxPYmT"  # Change this to a strong key
 
is_cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if is_cuda_available else "cpu")
print("✅ Is CUDA available?", is_cuda_available)
print("✅ CUDA Device Name:", torch.cuda.get_device_name(0) if is_cuda_available else "No GPU detected")
 
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True).to(device)
print("✅ Model is running on:", next(model.parameters()).device)
 
def load_image(image_path):
    if os.path.exists(image_path):  # Local file
        return Image.open(image_path).convert("RGB")
    parsed_url = urlparse(image_path)
    if parsed_url.scheme in ["http", "https"]:  # URL
        response = requests.get(image_path)
        return Image.open(BytesIO(response.content)).convert("RGB")
    raise ValueError(f"Invalid image path: {image_path}. Provide a valid file path or URL.")
 
def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]  # Convert grayscale to RGB
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)  # Convert to tensor
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor, 255.0)  # Normalize to [0,1]
    image = normalize(image, [0.5], [1.0])  # Normalize for model input
    image = image.to(device)  
    print("✅ Image tensor is on:", image.device)
    return image

# ✅ Postprocessing Function
def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
    result = (result - result.min()) / (result.max() - result.min())  # Normalize mask
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    return np.squeeze(im_array)  # Convert to NumPy array

# ✅ Remove Background API (Requires API Key)
@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    try:
        # ✅ Check for API Key in Headers
        api_key = request.headers.get("Authorization")
        if not api_key or api_key != f"Bearer {API_KEY}":
            return jsonify({"error": "Unauthorized. Invalid API key."}), 403

        # ✅ Process Image File Upload or URL
        if "file" in request.files:  # Image from form-data (File Upload)
            image = Image.open(request.files["file"]).convert("RGB")
        else:  # Image from JSON (URL)
            data = request.get_json()
            image_url = data.get("image_url")
            image = load_image(image_url)

        orig_im = np.array(image)
        orig_im_size = orig_im.shape[0:2]
        model_input_size = [1024, 1024]

        # ✅ Process Image
        image_tensor = preprocess_image(orig_im, model_input_size)

        # ✅ Run Model on GPU
        with torch.no_grad():
            result = model(image_tensor)  # ✅ Ensure tensor is on GPU
            print("✅ Result tensor is on:", result[0][0].device)

        # ✅ Generate Transparent Image
        result_image = postprocess_image(result[0][0], orig_im_size)
        mask_image = Image.fromarray(result_image)

        no_bg_image = image.copy()
        no_bg_image.putalpha(mask_image)  # Apply alpha transparency

        # ✅ Save and Return the Result
        img_io = BytesIO()
        no_bg_image.save(img_io, format="PNG")
        img_io.seek(0)

        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
