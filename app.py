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
        "font": "Arial",
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
    # Додаткові стилі субтитрів...
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

# --- Функція для пошуку схожих зображень за текстом ---
def search_images(query):
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": 5}  # Зменшено кількість для кращої продуктивності
    response = requests.get(pexels_url, headers=headers, params=params)
    data = response.json()

    # Якщо немає результатів, використовуємо зображення-заглушку
    if "photos" not in data or len(data["photos"]) == 0:
        return ["https://via.placeholder.com/1280x720.png?text=No+Image"]
    
    # Повертаємо URL першого зображення
    return [photo["src"]["landscape"] for photo in data["photos"]]

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
        img_clips = []

        for segment in lines:
            # Пошук схожих зображень за першим словом в сегменті
            query = segment.text.split()[0] if segment.text else "nature"
            img_urls = search_images(query)

            for img_url in img_urls:
                try:
                    img_data = requests.get(img_url).content
                    img = Image.open(BytesIO(img_data)).resize((1280, 720))
                    img = np.array(img)  # Перетворюємо PIL об'єкт у numpy масив
                except Exception as e:
                    st.warning(f"Помилка при обробці зображення: {e}")
                    img = np.zeros((720, 1280, 3), dtype=np.uint8)  # Чорний фон як заглушка

                duration = segment.end - segment.start
                img_clip = ImageClip(img).set_duration(duration)

                # Створення субтитрів
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
