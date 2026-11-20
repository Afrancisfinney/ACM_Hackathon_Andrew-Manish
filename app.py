import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS so your frontend webpage can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. CNN Architecture matching your trained model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# 2. Load model and weights
device = torch.device("cpu")
num_classes = 10  # c0 through c9
model = SimpleCNN(num_classes=num_classes)

try:
    model.load_state_dict(torch.load('model_weights.pth', map_location=device))
    print("Loaded 'model_weights.pth' successfully!")
except Exception as e:
    print(f"Warning: Could not load 'model_weights.pth' ({e}). Make sure you saved your model weights!")

model.eval()

# 3. State Farm Driver Classes
class_names = {
    0: "Safe Driving (c0)",
    1: "Texting - Right (c1)",
    2: "Talking on phone - Right (c2)",
    3: "Texting - Left (c3)",
    4: "Talking on phone - Left (c4)",
    5: "Operating radio (c5)",
    6: "Drinking (c6)",
    7: "Reaching behind (c7)",
    8: "Hair and makeup (c8)",
    9: "Talking to passenger (c9)"
}

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    class_id = predicted_idx.item()
    class_label = class_names.get(class_id, "Unknown")
    is_distracted = False if class_id == 0 else True
    
    return {
        "class_id": class_id,
        "prediction": class_label,
        "is_distracted": is_distracted,
        "confidence": round(confidence.item() * 100, 2)
    }
