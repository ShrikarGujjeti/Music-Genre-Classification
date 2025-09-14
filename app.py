import os
import tempfile
import pickle

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

st.set_page_config(page_title="Music Genre Classification", layout="wide")

# --- Header ---
st.title("Music Genre Classification")
st.write("Upload a WAV file (short clips recommended). The app shows simple visualizations and a classical ML prediction based on MFCCs.")

# --- Sidebar ---
with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
    st.markdown("## Model")
    st.markdown("- Classical ML model using MFCC mean features\n- Scaler required (scaler.pkl)")
    st.markdown("## Notes")
    st.markdown("- Best with clean short clips (≤ 15s)")

# --- Load model and scaler ---
ml_model = None
scaler = None
try:
    with open("music_genre_model.pkl", "rb") as f:
        ml_model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error("Model or scaler not found or failed to load. Place music_genre_model.pkl and scaler.pkl next to this app.")
    st.stop()

if uploaded_file is None:
    st.info("Please upload a WAV file to continue.")
    st.stop()

# --- Process uploaded file ---
try:
    # Keep a copy for librosa
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Audio preview (Streamlit can play the original uploaded bytes)
    st.audio(uploaded_file)

    # Load audio (limit for speed)
    audio, sr = librosa.load(tmp_path, duration=15)

    # Features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    # --- Plots ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Waveform")
        fig_wf, ax_wf = plt.subplots(figsize=(6,2.5))
        times = np.arange(len(audio)) / float(sr)
        ax_wf.plot(times, audio, color="#0b5cff", linewidth=0.6)
        ax_wf.set_xlabel("Time (s)")
        ax_wf.set_ylabel("Amplitude")
        ax_wf.set_xlim(0, times.max() if times.size else 1)
        ax_wf.grid(alpha=0.2)
        st.pyplot(fig_wf)
        plt.close(fig_wf)

        st.subheader("Prediction")
        mfcc_mean = np.mean(mfcc.T, axis=0)
        X = scaler.transform([mfcc_mean])
        pred = ml_model.predict(X)[0]
        st.success(f"Predicted genre: {pred}")

        if hasattr(ml_model, "predict_proba"):
            probs = ml_model.predict_proba(X)[0]
            classes = getattr(ml_model, "classes_", None)
            if classes is not None:
                top_idx = np.argsort(probs)[::-1][:3]
                st.write("Top probabilities:")
                for i in top_idx:
                    st.write(f"- {classes[i]}: {probs[i]*100:.1f}%")

    with col2:
        st.subheader("Mel Spectrogram")
        fig_spec, ax_spec = plt.subplots(figsize=(6,2.5))
        img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax_spec, cmap="magma")
        ax_spec.set_title("Mel Spectrogram (dB)")
        fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB", fraction=0.046)
        st.pyplot(fig_spec)
        plt.close(fig_spec)

        st.subheader("MFCC (heatmap)")
        fig_mfcc, ax_mfcc = plt.subplots(figsize=(6,2.5))
        img2 = ax_mfcc.imshow(mfcc, aspect="auto", origin="lower", cmap="viridis")
        ax_mfcc.set_xlabel("Frames")
        ax_mfcc.set_ylabel("MFCC coeff")
        ax_mfcc.set_title("MFCC")
        fig_mfcc.colorbar(img2, ax=ax_mfcc, fraction=0.046)
        st.pyplot(fig_mfcc)
        plt.close(fig_mfcc)

    # cleanup
    try:
        os.remove(tmp_path)
    except OSError:
        pass

except Exception as e:
    st.error(f"Failed to process audio: {e}")
