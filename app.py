import streamlit as st
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, TextClip, CompositeVideoClip
from faster_whisper import WhisperModel
from pydub import AudioSegment
import requests
from io import BytesIO
import os
from PIL import Image
import numpy as np
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
st.markdown("### 🎨 Виберіть стиль субтитрів")

# Список шаблонів субтитрів
subtitle_styles = {
    "Класичний білий": {
        "color": "FFFFFF", "bg_color": "000000", "font_size": 40, "font": "Arial", "image": "https://via.placeholder.com/640x360.png?text=Classic+White"
    },
    "Темний з жовтим текстом": {
        "color": "FFD700", "bg_color": "000000", "font_size": 42, "font": "Arial-Bold", "image": "https://via.placeholder.com/640x360.png?text=Dark+Yellow"
    },
    "Рожевий глянець": {
        "color": "FF69B4", "bg_color": "1A1A1A", "font_size": 38, "font": "Georgia", "image": "https://via.placeholder.com/640x360.png?text=Pink+Glossy"
    },
    "Контрастний біло-червоний": {
        "color": "FFFFFF", "bg_color": "B22222", "font_size": 45, "font": "Impact", "image": "https://via.placeholder.com/640x360.png?text=White+Red"
    },
    "М’який синій": {
        "color": "ADD8E6", "bg_color": "2C3E50", "font_size": 36, "font": "Tahoma", "image": "https://via.placeholder.com/640x360.png?text=Soft+Blue"
    },
    "Помаранчевий кінотеатр": {
        "color": "FFA500", "bg_color": "000000", "font_size": 50, "font": "Helvetica-Bold", "image": "https://via.placeholder.com/640x360.png?text=Cinema+Orange"
    },
    "Футуристичний зелений": {
        "color": "00FF7F", "bg_color": "101010", "font_size": 42, "font": "Courier-New", "image": "https://via.placeholder.com/640x360.png?text=Futuristic+Green"
    },
    "Стиль Netflix": {
        "color": "FFFFFF", "bg_color": "000000", "font_size": 48, "font": "Verdana-Bold", "image": "https://via.placeholder.com/640x360.png?text=Netflix+Style"
    },
    "Сучасний біло-сірий": {
        "color": "F0F0F0", "bg_color": "333333", "font_size": 40, "font": "Arial", "image": "https://via.placeholder.com/640x360.png?text=Modern+Gray"
    },
    "Журналний стиль": {
        "color": "000000", "bg_color": "FFD700", "font_size": 46, "font": "Times-New-Roman", "image": "https://via.placeholder.com/640x360.png?text=Magazine+Style"
    }
}

# Вибір шаблону
selected_style = st.selectbox("Оберіть шаблон субтитрів:", list(subtitle_styles.keys()))
style = subtitle_styles[selected_style]

# Відображення картки зображення шаблону
st.image(style["image"], caption=f"Приклад субтитрів: {selected_style}", use_column_width=True)

# --- Кнопка генерації відео ---
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
                img = np.array(img)  # Перетворюємо PIL об'єкт у numpy масив
            except Exception as e:
                st.warning(f"Помилка при обробці зображення: {e}")
                img = np.zeros((720, 1280, 3), dtype=np.uint8)  # Чорний фон як заглушка

            duration = segment.end - segment.start
            img_clip = ImageClip(img).set_duration(duration)

            # --- Субтитри ---
            txt_clip = TextClip(
                segment.text,
                fontsize=style["font_size"],
                font=style["font"],
                color=style["color"],
                bg_color=style["bg_color"],
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
