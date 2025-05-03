import streamlit as st
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, TextClip, CompositeVideoClip
from faster_whisper import WhisperModel
from pydub import AudioSegment
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import tempfile

# --- Константи ---
PEXELS_API_KEY = st.secrets["PEXELS_API_KEY"]
pexels_url = "https://api.pexels.com/v1/search"

# --- СТИЛІ СУБТИТРІВ ---
subtitle_styles = {
    "Класичний білий": {
        "font": "Arial-Bold",
        "fontsize": 48,
        "color": "white",
        "stroke_color": "black",
        "stroke_width": 2,
        "image": "https://via.placeholder.com/640x100.png?text=Класичний+білий"
    },
    "Жовтий з тінню": {
        "font": "Arial-Bold",
        "fontsize": 46,
        "color": "yellow",
        "stroke_color": "black",
        "stroke_width": 3,
        "image": "https://via.placeholder.com/640x100.png?text=Жовтий+з+тінню"
    },
    "Червоний жирний": {
        "font": "Arial-Bold",
        "fontsize": 50,
        "color": "red",
        "stroke_color": "black",
        "stroke_width": 2,
        "image": "https://via.placeholder.com/640x100.png?text=Червоний+жирний"
    },
    "Зелений тонкий": {
        "font": "Arial",
        "fontsize": 44,
        "color": "green",
        "stroke_color": "black",
        "stroke_width": 1,
        "image": "https://via.placeholder.com/640x100.png?text=Зелений+тонкий"
    },
    "Помаранчевий без тіні": {
        "font": "Arial-Bold",
        "fontsize": 48,
        "color": "orange",
        "stroke_color": None,
        "stroke_width": 0,
        "image": "https://via.placeholder.com/640x100.png?text=Помаранчевий+простий"
    },
    "Блакитний глянець": {
        "font": "Arial-Bold",
        "fontsize": 50,
        "color": "cyan",
        "stroke_color": "blue",
        "stroke_width": 2,
        "image": "https://via.placeholder.com/640x100.png?text=Блакитний+глянець"
    },
    "Чорно-білий контраст": {
        "font": "Arial-Bold",
        "fontsize": 48,
        "color": "white",
        "stroke_color": "black",
        "stroke_width": 4,
        "image": "https://via.placeholder.com/640x100.png?text=Контраст+ч/б"
    },
    "Фіолетовий стилізований": {
        "font": "Arial-Bold",
        "fontsize": 45,
        "color": "purple",
        "stroke_color": "black",
        "stroke_width": 2,
        "image": "https://via.placeholder.com/640x100.png?text=Фіолетовий+стиль"
    },
    "Сірий мінімал": {
        "font": "Arial",
        "fontsize": 42,
        "color": "gray",
        "stroke_color": None,
        "stroke_width": 0,
        "image": "https://via.placeholder.com/640x100.png?text=Сірий+мінімал"
    },
    "Яскраво-рожевий": {
        "font": "Arial-Bold",
        "fontsize": 46,
        "color": "deeppink",
        "stroke_color": "black",
        "stroke_width": 2,
        "image": "https://via.placeholder.com/640x100.png?text=Рожевий+ефект"
    },
}

# --- ІНТЕРФЕЙС ---
st.set_page_config(page_title="Автовідео зі сценарію", layout="centered")
st.title("🎬 Автоматичне відео з озвучкою та субтитрами")

script = st.text_area("📝 Введіть текст сценарію", height=200)

# Вибір шаблону
selected_style = st.selectbox("🎨 Оберіть стиль субтитрів", list(subtitle_styles.keys()))
style = subtitle_styles[selected_style]

# Відображення картинки для вибраного шаблону
st.image(style["image"], caption=f"Приклад: {selected_style}", use_container_width=True)

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
                img_data = requests.get(img_url).content
                img = Image.open(BytesIO(img_data)).resize((1280, 720))
            except Exception:
                img = Image.new('RGB', (1280, 720), color=(73, 109, 137))

            duration = segment.end - segment.start
            img_clip = ImageClip(img).set_duration(duration)

            # Створення субтитрів
            try:
                txt_clip = TextClip(
                    segment.text,
                    fontsize=style["fontsize"],
                    font=style["font"],
                    color=style["color"],
                    stroke_color=style["stroke_color"],
                    stroke_width=style["stroke_width"],
                    method='caption',
                    size=(1200, None),
                ).set_duration(duration).set_position(("center", "bottom"))
            except Exception as e:
                st.warning(f"⚠️ Помилка при генерації субтитру: {e}")
                txt_clip = None

            final_clip = CompositeVideoClip([img_clip, txt_clip]) if txt_clip else img_clip
            img_clips.append(final_clip)

    with st.spinner("🎞️ Монтуємо відео..."):
        final_video = concatenate_videoclips(img_clips, method="compose")
        final_video = final_video.set_audio(AudioFileClip(temp_audio.name))

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        final_video.write_videofile(output_path, fps=24)

    st.success("✅ Відео готове!")
    st.video(output_path)

else:
    st.info("⬆️ Введіть сценарій та натисніть кнопку для генерації.")
