from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io
import os
import traceback

# ------------------
# Torch safety for Render
# ------------------
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ------------------
# FastAPI App
# ------------------
app = FastAPI()

# ------------------
# CORS Middleware
# ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------
# CNN Feature Extractor (FIXED)
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

            # 🔥 GUARANTEES FIXED SHAPE
            nn.AdaptiveAvgPool2d((16, 64)),
            nn.Flatten()
        )

        self.fc = nn.Linear(64 * 16 * 64, out_dim)

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)

# ------------------
# Personality Predictor
# ------------------
class PersonalityPredictor(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.model(x)

# ------------------
# Load Models (ONCE)
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
# Image Transform (FIXED)
# ------------------
transform = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor(),          # produces [1, 64, 256]
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ------------------
# Health Check (IMPORTANT)
# ------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ------------------
# Predict Endpoint (SAFE)
# ------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()

        # Load image safely
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            features = cnn(img)
            output = predictor(features)
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

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
