import streamlit as st
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from faster_whisper import WhisperModel
from pydub import AudioSegment
import requests
from io import BytesIO
from PIL import Image
import tempfile
from gtts import gTTS
import random

# --- Константи ---
PEXELS_API_KEY = st.secrets["PEXELS_API_KEY"]
pexels_url = "https://api.pexels.com/v1/search"
HEADERS = {"Authorization": PEXELS_API_KEY}

st.set_page_config(page_title="Відеогенератор", layout="centered")
st.title("🎬 Автоматичне відео з озвучкою та фоновими зображеннями")

# --- Ввід сценарію ---
script = st.text_area("📝 Введіть текст сценарію", height=200)

# --- Кнопка ---
if st.button("🎥 Згенерувати відео") and script.strip() != "":
    with st.spinner("🔊 Створюємо озвучку..."):
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
            keywords = segment.text.lower().split()
            search_terms = random.sample(keywords, min(3, len(keywords))) if keywords else ["nature"]
            query = " ".join(search_terms)

            params = {"query": query, "per_page": 1}
            try:
                response = requests.get(pexels_url, headers=HEADERS, params=params, timeout=10)
                data = response.json()
                img_url = data["photos"][0]["src"]["landscape"]
                img_data = requests.get(img_url, timeout=10).content
                img = Image.open(BytesIO(img_data)).resize((1280, 720))
            except Exception:
                img = Image.new('RGB', (1280, 720), color=(40, 40, 40))

            duration = segment.end - segment.start
            clip = ImageClip(img).set_duration(duration)
            img_clips.append(clip)

    with st.spinner("🎞️ Монтуємо відео..."):
        final_video = concatenate_videoclips(img_clips, method="compose")
        final_video = final_video.set_audio(AudioFileClip(temp_audio.name))

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        final_video.write_videofile(output_path, fps=24)

    st.success("✅ Відео згенеровано!")
    st.video(output_path)

else:
    st.info("⬆️ Введіть текст сценарію та натисніть кнопку.")
