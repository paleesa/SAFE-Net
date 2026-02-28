import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

class AgeResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet50(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

class AgePredictor:
    def __init__(self, ckpt_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AgeResNet50().to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.tfm = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    @torch.no_grad()
    def predict_age(self, img: Image.Image) -> float:
        img = img.convert("RGB")
        x = self.tfm(img).unsqueeze(0).to(self.device)
        pred = self.model(x).item()
        pred = max(0.0, min(116.0, pred))
        return pred