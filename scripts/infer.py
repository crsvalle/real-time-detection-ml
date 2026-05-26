"""
infer.py — run vehicle make/model/year inference on a single image or crop.

Usage:
  python infer.py path/to/image.jpg

Returns JSON to stdout — designed to be called from a backend API endpoint
that your React frontend can hit instead of (or before) Claude Vision.
"""

import sys
import json
import re
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH  = Path("car_classifier_best.pth")
CLASSES_PATH = Path("car_classes.json")
TOP_K = 3  # return top-3 predictions so the frontend can show alternatives

# -------------------------
# TRANSFORMS (must match val_transform in train.py)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------------
# LOAD MODEL + CLASS NAMES (cached at module level for server use)
# -------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

with open(CLASSES_PATH) as f:
    class_names = json.load(f)  # e.g. ["AM General Hummer SUV 2000", ...]

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 196)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.to(device)


def parse_class_name(raw: str) -> dict:
    """
    Parse a Stanford Cars class string like "2012 Toyota Camry Sedan"
    into { brand, model, year, body_style }.
    """
    # Year is always the last token and 4 digits
    match = re.match(r"(\d{4})\s+(.+)", raw)
    if match:
        year = match.group(1)
        rest = match.group(2)
    else:
        year = "Unknown"
        rest = raw

    # Body style is the last word if it's a known type
    BODY_STYLES = {"Sedan", "SUV", "Coupe", "Convertible", "Wagon",
                   "Pickup", "Van", "Hatchback", "Cab"}
    parts = rest.split()
    if parts and parts[-1] in BODY_STYLES:
        body_style = parts[-1]
        rest = " ".join(parts[:-1])
    else:
        body_style = ""

    # First word is brand, rest is model
    parts = rest.split(maxsplit=1)
    brand = parts[0] if parts else "Unknown"
    model_name = parts[1] if len(parts) > 1 else ""

    return {
        "brand": brand,
        "model": model_name,
        "year": year,
        "body_style": body_style,
    }


def predict(image_path: str) -> dict:
    """
    Run inference on an image file.
    Returns a dict compatible with VehicleInfoPanel props.
    """
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, TOP_K)

    top_prob = top_probs[0].item()
    top_idx  = top_indices[0].item()
    raw_name = class_names[top_idx]

    parsed = parse_class_name(raw_name)

    # Map confidence score to low/medium/high label
    if top_prob >= 0.70:
        confidence = "high"
    elif top_prob >= 0.40:
        confidence = "medium"
    else:
        confidence = "low"

    alternatives = []
    for prob, idx in zip(top_probs[1:], top_indices[1:]):
        alt = parse_class_name(class_names[idx.item()])
        alternatives.append({**alt, "score": round(prob.item(), 3)})

    return {
        "brand":       parsed["brand"],
        "model":       f"{parsed['model']} {parsed['body_style']}".strip(),
        "year":        parsed["year"],
        "confidence":  confidence,
        "notes":       f"Score: {top_prob:.1%}. Alt: {alternatives[0]['brand']} {alternatives[0]['model']}",
        "raw_class":   raw_name,
        "score":       round(top_prob, 4),
        "alternatives": alternatives,
    }


# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py <image_path>")
        sys.exit(1)

    result = predict(sys.argv[1])
    print(json.dumps(result, indent=2))