from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io
import os

# ------------------
# FastAPI App
# ------------------
app = FastAPI()

# ------------------
# CORS Middleware
# ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------
# CNN Feature Extractor
# ------------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )
        # After resize (64, 256) → (64, 16, 64)
        self.fc = nn.Linear(64 * 16 * 64, out_dim)

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)

# ------------------
# Personality Predictor (CNN ONLY)
# ------------------
class PersonalityPredictor(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5)   # Big Five traits
        )

    def forward(self, x):
        return self.model(x)

# ------------------
# Load Models
# ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cnn = CNNFeatureExtractor()
cnn.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "models/cnn_model.pth"), map_location="cpu")
)
cnn.eval()

predictor = PersonalityPredictor()
predictor.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "models/personality_predictor.pth"), map_location="cpu")
)
predictor.eval()

# ------------------
# Image Transform
# ------------------
transform = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ------------------
# Predict Endpoint
# ------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()

    # Load and preprocess image
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = transform(img).unsqueeze(0).float()

    with torch.no_grad():
        features = cnn(img)
        output = predictor(features)

        # Sigmoid for independent Big-5 scores
        scores = torch.sigmoid(output).squeeze().tolist()

    traits = [
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism"
    ]

    dominant_index = scores.index(max(scores))

    response = dict(zip(traits, scores))
    response["dominant_trait"] = traits[dominant_index]

    return response
