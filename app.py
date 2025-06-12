import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Page Title
st.set_page_config(page_title="Rainy Day Visualizer", layout="centered")
st.title("🎧 Embedded Visualizer")

# Audio Card UI Mockup (Styled HTML)
st.components.v1.html("""
<div style="background: linear-gradient(to right, #6a737b, #9aa2a9); 
            border-radius: 15px; padding: 20px; color: white;
            width: 100%; max-width: 500px; margin: auto; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.3); font-family: sans-serif;">
    <div style="font-size: 14px; background: white; color: #555; padding: 5px 15px;
                border-radius: 20px; width: fit-content; margin-bottom: 15px;">
        <b>Now Playing</b>
    </div>
    <h2 style="margin: 0 0 5px;">Rainy Day</h2>
    <p style="margin: 0; font-size: 14px; letter-spacing: 2px;">AARON LOEB</p>
    <div style="margin-top: 20px; display: flex; gap: 4px; align-items: flex-end; height: 40px;">
        <div style="width: 4px; height: 12px; background: white; border-radius: 2px;"></div>
        <div style="width: 4px; height: 20px; background: white; border-radius: 2px;"></div>
        <div style="width: 4px; height: 8px; background: white; border-radius: 2px;"></div>
        <div style="width: 4px; height: 16px; background: white; border-radius: 2px;"></div>
        <div style="width: 4px; height: 26px; background: white; border-radius: 2px;"></div>
        <div style="width: 4px; height: 8px; background: white; border-radius: 2px;"></div>
        <div style="width: 4px; height: 14px; background: white; border-radius: 2px;"></div>
    </div>
</div>
""", height=200)

# Upload WAV File
st.subheader("🎵 Upload Your Own WAV File")
uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])

if uploaded_file:
    # Load and play audio
    y, sr = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file, format='audio/wav')

    # Duration & Tempo
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    st.markdown(f"**Duration:** {duration:.2f} seconds")
    st.markdown(f"**Estimated Tempo:** {tempo:.2f} BPM")

    # Plot Waveform
    st.subheader("📈 Waveform")
    fig1, ax1 = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='dodgerblue')
    ax1.set(title='Waveform')
    st.pyplot(fig1)

    # Plot Spectrogram
    st.subheader("🌈 Spectrogram")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig2, ax2 = plt.subplots()
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.f dB")
    ax2.set(title='Spectrogram')
    st.pyplot(fig2)
else:
    st.info("Please upload a WAV file to begin visualization.")
