import streamlit as st
import cv2
import numpy as np
import joblib

from skimage.feature import hog
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]

# =========================
# LOAD MODEL
# =========================
scaler = joblib.load("saved_model_final/scaler.pkl")
pca = joblib.load("saved_model_final/pca.pkl")
models = joblib.load("saved_model_final/subspace_knn.pkl")

mobilenet = load_model("saved_model_final/mobilenet.h5")

# =========================
# FEATURE EXTRACTION
# =========================
def extract_hog(img):
    return hog(img, pixels_per_cell=(32, 32), cells_per_block=(2, 2))

def extract_mobilenet(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, IMG_SIZE)
    rgb = preprocess_input(rgb.astype("float32"))
    feat = mobilenet.predict(np.expand_dims(rgb, 0), verbose=0)
    return feat[0]

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feat = extract_hog(gray)
    mob_feat = extract_mobilenet(gray)

    features = np.concatenate([hog_feat, mob_feat]).reshape(1, -1)

    # preprocessing
    features = scaler.transform(features)
    features = pca.transform(features)

    # subspace kNN voting
    preds = np.array([m.predict(features[:, idx]) for m, idx in models])
    final = np.bincount(preds.flatten()).argmax()

    return CLASSES[final]

# =========================
# UI STREAMLIT
# =========================
st.set_page_config(page_title="Brain Tumor Classification AI", layout="centered")

st.title("🧠 Brain Tumor Classification System")
st.write("Upload MRI image untuk prediksi jenis tumor")

uploaded_file = st.file_uploader("Pilih gambar MRI", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR", caption="Input MRI Image")

    if st.button("Predict"):

        result = predict_image(img)

        st.success(f"Hasil Prediksi: **{result}**")

        st.write("Model: HOG + MobileNet + PCA + Subspace kNN")