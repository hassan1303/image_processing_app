
import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="🖼️ Image Processing Playground", layout="wide")
st.title("🖼️ Image Processing Playground")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img_np = np.array(img)
    st.sidebar.title("🔧 Image Processing Techniques")
    processing_option = st.sidebar.selectbox("Select Technique", [
        "Thresholding",
        "Blurring & Smoothing",
        "Edge Detection",
        "Contour Detection",
        "Color Space Conversion"
    ])

    def to_grayscale_safe(image):
        return image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def apply_thresholding(image):
        gray = to_grayscale_safe(image)
        method = st.sidebar.radio("Thresholding Method", ["Binary", "Otsu", "Adaptive"])
        thresh_val = st.sidebar.slider("Threshold Value", 0, 255, 127)

        if method == "Binary":
            _, result = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        elif method == "Otsu":
            _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        return result

    def apply_blur(image):
        method = st.sidebar.radio("Blur Type", ["Averaging", "Gaussian", "Median"])
        ksize = st.sidebar.slider("Kernel Size", 1, 15, 5, step=2)
        if method == "Averaging":
            return cv2.blur(image, (ksize, ksize))
        elif method == "Gaussian":
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        else:
            return cv2.medianBlur(image, ksize)

    def apply_edge_detection(image):
        method = st.sidebar.radio("Edge Detection Method", ["Canny", "Sobel", "Laplacian"])
        gray = to_grayscale_safe(image)
        if method == "Canny":
            low = st.sidebar.slider("Low Threshold", 0, 255, 50)
            high = st.sidebar.slider("High Threshold", 0, 255, 150)
            return cv2.Canny(gray, low, high)
        elif method == "Sobel":
            return cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        else:
            return cv2.Laplacian(gray, cv2.CV_64F)

    def apply_contour_detection(image):
        gray = to_grayscale_safe(image)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        output = image.copy()
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
        return output

    def apply_color_conversion(image):
        option = st.sidebar.selectbox("Color Space", ["Grayscale", "HSV", "BGR"])
        if option == "Grayscale":
            return to_grayscale_safe(image)
        elif option == "HSV":
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        else:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    result = None
    if processing_option == "Thresholding":
        result = apply_thresholding(img_np)
    elif processing_option == "Blurring & Smoothing":
        result = apply_blur(img_np)
    elif processing_option == "Edge Detection":
        result = apply_edge_detection(img_np)
    elif processing_option == "Contour Detection":
        result = apply_contour_detection(img_np)
    elif processing_option == "Color Space Conversion":
        result = apply_color_conversion(img_np)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_np, caption="Original Image", use_column_width=True)
    with col2:
        if result is not None:
            st.image(result, caption="Processed Image", use_column_width=True, 
                     channels="GRAY" if len(result.shape) == 2 else "RGB")
