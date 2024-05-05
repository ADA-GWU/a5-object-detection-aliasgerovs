import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import easyocr
import matplotlib.pyplot as plt

model_weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=model_weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model.eval()

if torch.cuda.is_available():
    model = model.cuda()

def preprocess_image_for_cnn(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_with_cnn(image, model):
    image_tensor = preprocess_image_for_cnn(image) 
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
    return preds.item() 

# GUI Setup
refPt = []
cropping = False

def click_and_select_corners(event, x, y, flags, param):
    global refPt, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        cropping = False
        cv2.circle(image, refPt[-1], 5, (0, 255, 0), -1)
        cv2.imshow("image", image)
        if len(refPt) == 4:
            cv2.putText(image, 'Press "p" to process', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("image", image)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# Load and process image
image = cv2.imread('receipt.jpeg')
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_select_corners)

while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("p") and len(refPt) == 4:
        break

# Apply the perspective transformation
warped = four_point_transform(image, refPt)
cv2.imshow("Warped Image", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# OCR setup
mser = cv2.MSER_create(delta=2, min_area=10, max_area=500)
gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
regions, _ = mser.detectRegions(gray)
predicted_characters = []
reader = easyocr.Reader(['en'], gpu=True)

def predict_with_easyocr(image):
    results = reader.readtext(image)
    return " ".join([result[1] for result in results])

# Visualization
fig, axs = plt.subplots(7, 7, figsize=(15, 15))
axs = axs.flatten()
count = 0

for region in regions:
    x, y, w, h = cv2.boundingRect(region)
    roi = gray[y:y+h, x:x+w]
    if roi.size == 0:
        continue
    roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    predicted_text = predict_with_easyocr(roi_color)
    predicted_characters.append(predicted_text)
    cv2.rectangle(warped, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if predicted_text and count < 49:
        axs[count].imshow(roi_color, cmap='gray')
        axs[count].set_title(f"Detected: {predicted_text}", fontsize=8)
        axs[count].axis('off')
        count += 1

cv2.imshow("Detected", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

for ax in axs[count:]:
    ax.axis('off')

plt.tight_layout()
plt.show()

print("Predicted:", " ".join(predicted_characters))