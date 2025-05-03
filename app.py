import streamlit as st
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, TextClip, CompositeVideoClip
from faster_whisper import WhisperModel
from pydub import AudioSegment
import requests
from io import BytesIO
import os
from PIL import Image
import tempfile

# --- Константи ---
PEXELS_API_KEY = st.secrets["PEXELS_API_KEY"]
pexels_url = "https://api.pexels.com/v1/search"

# --- СТИЛІ САЙТУ ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', sans-serif;
        color: #333;
    }
    .css-1v3fvcr { padding: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Автоматичне відео зі сценарію")
st.subheader("📝 Введіть сценарій — отримаєте змонтоване відео зі зображеннями та субтитрами")

# Ввід тексту
script = st.text_area("Введіть текст сценарію", height=200)

# --- СТИЛІ СУБТИТРІВ ---
st.markdown("### 🎨 Стиль субтитрів")
subtitle_color = st.color_picker("Колір тексту", "#FFFFFF")
bg_color = st.color_picker("Колір фону тексту", "#000000")
font_size = st.slider("Розмір шрифту", 20, 70, 40)

# --- Кнопка ---
if st.button("🎥 Згенерувати відео") and script.strip() != "":
    with st.spinner("🔊 Генеруємо озвучку..."):
        from gtts import gTTS
        tts = gTTS(script, lang="uk")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)

        audio = AudioSegment.from_mp3(temp_audio.name)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(temp_wav.name, format="wav")

    with st.spinner("🧠 Визначаємо таймінги..."):
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(temp_wav.name, beam_size=5)
        lines = list(segments)

    with st.spinner("🖼️ Підбираємо зображення..."):
        headers = {"Authorization": PEXELS_API_KEY}
        img_clips = []

        for segment in lines:
            query = segment.text.split()[0] if segment.text else "nature"
            params = {"query": query, "per_page": 1}
            response = requests.get(pexels_url, headers=headers, params=params)
            data = response.json()

            try:
                img_url = data["photos"][0]["src"]["landscape"]
            except Exception as e:
                st.warning(f"Помилка при завантаженні зображення: {e}")
                img_url = "https://via.placeholder.com/1280x720.png?text=No+Image"

            try:
                img_data = requests.get(img_url).content
                img = Image.open(BytesIO(img_data)).resize((1280, 720))
            except Exception as e:
                st.warning(f"Помилка при обробці зображення: {e}")
                img = Image.new('RGB', (1280, 720), color=(73, 109, 137))

            duration = segment.end - segment.start
            img_clip = ImageClip(img).set_duration(duration)

            # --- Субтитри ---
            txt_clip = TextClip(
                segment.text,
                fontsize=font_size,
                font="Arial",
                color=subtitle_color.replace("#", ""),
                bg_color=bg_color.replace("#", ""),
                size=(1200, None),
                method='caption'
            ).set_position(("center", "bottom")).set_duration(duration)

            video_clip = CompositeVideoClip([img_clip, txt_clip])
            img_clips.append(video_clip)

    with st.spinner("🎞️ Монтуємо відео..."):
        final_video = concatenate_videoclips(img_clips, method="compose")
        final_video = final_video.set_audio(AudioFileClip(temp_audio.name))

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        final_video.write_videofile(output_path, fps=24)

    st.success("✅ Відео згенеровано!")
    st.video(output_path)
else:
    st.info("⬆️ Введіть текст сценарію і натисніть кнопку.")
