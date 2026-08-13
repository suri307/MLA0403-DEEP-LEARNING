import cv2
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/Surendhar Reddy/Downloads/MAX VERSTAPPEN.jpg", 0)

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(thresh, cmap='gray')
plt.title("Threshold")

plt.subplot(1,3,3)
plt.imshow(closing, cmap='gray')
plt.title("Segmented")

plt.show()
