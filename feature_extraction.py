import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

CROP_FOLDER = 'crops/broadcast'   # Change to crops/tacticam for tacticam crops
FEATURES_SAVE_PATH = 'features_broadcast.npy'  # Change for tacticam

# ---- Load Pretrained ResNet50 ----
resnet = models.resnet50(weights='IMAGENET1K_V2')
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove final classification layer
resnet.eval()

# ---- Preprocessing ----
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---- Extract Features ----
features = []
image_names = []

for filename in tqdm(sorted(os.listdir(CROP_FOLDER))):
    if not filename.endswith('.jpg'):
        continue

    img_path = os.path.join(CROP_FOLDER, filename)
    image = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        feat = resnet(img_tensor).squeeze().numpy()  # Shape: (2048,)

    features.append(feat)
    image_names.append(filename)

features = np.array(features)  # Shape: (num_images, 2048)
np.save(FEATURES_SAVE_PATH, {'features': features, 'filenames': image_names})

print(f"Saved features to {FEATURES_SAVE_PATH}")
