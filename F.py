import cv2
import torch
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import seaborn as sns
from PIL import Image, ImageTk

def load_image(loc):
    img = cv2.imread(str(loc))
    if img is None:
        raise FileNotFoundError(f"Image not found at {loc}")
    return img

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def show_image(img, title=''):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = ImageTk.PhotoImage(img)

    panel = tk.Label(root, image=img)
    panel.image = img
    panel.pack(pady=10)
    panel.title(title)
    panel.axis('off')

def calc_histogram(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist.flatten()

def gray2bw(img, threshold=100):
    _, bw_image = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return bw_image

def show_seaborn_histogram(img):
    flat_img = img.flatten()
    sns.histplot(flat_img, bins=256, kde=False)
    plt.title('Seaborn Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Image Processing with PyTorch, Tkinter, and Seaborn")

    image_path = Path("D:/Another/Image/image_/image_1.jpg")

    # Load the image
    original_image = load_image(image_path)

    # Show the original image
    show_image(original_image, title='Original Image')

    # Calculate and display the histogram
    histogram = calc_histogram(original_image)
    plt.plot(histogram)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

    # Convert the image to grayscale
    grayscale_image = rgb2gray(original_image)
    show_image(grayscale_image, title='Grayscale Image')

    # Convert the grayscale image to black and white
    threshold_value = 100
    bw_image = gray2bw(grayscale_image, threshold=threshold_value)
    show_image(bw_image, title='Black & White Image')

    # Show Seaborn Histogram
    show_seaborn_histogram(grayscale_image)

    root.mainloop()
