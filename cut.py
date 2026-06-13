import cv2

# Settings
INPUT_FILE = 'Picture3_orig.png'
OUTPUT_FILE = 'cropped_center.png'

# Load
img = cv2.imread(INPUT_FILE)
h, w = img.shape[:2]

# Calculate center half (0.25 to 0.75 on both axes)
# For 512x512, this is 128 to 384
y1, y2 = h // 4, 3 * h // 4
x1, x2 = w // 4, 3 * w // 4

# Crop and Save
cropped = img[y1:y2, x1:x2]
cv2.imwrite(OUTPUT_FILE, cropped)

print(f"Cropped {w}x{h} to {x2-x1}x{y2-y1}")