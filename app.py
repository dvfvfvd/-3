import streamlit as st
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from faster_whisper import WhisperModel
from pydub import AudioSegment
import requests
from io import BytesIO
import os
from PIL import Image
import tempfile

PEXELS_API_KEY = st.secrets["PEXELS_API_KEY"]
pexels_url = "https://api.pexels.com/v1/search"

st.title("🎬 Автоматичне відео з озвучкою")

# Ввід сценарію
script = st.text_area("Введіть текст сценарію", height=200)

# Кнопка генерації
if st.button("🎥 Згенерувати відео") and script.strip() != "":
    with st.spinner("🔊 Генеруємо озвучку..."):
        # Генеруємо тимчасовий аудіофайл
        from gtts import gTTS
        tts = gTTS(script, lang="uk")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)

        # Конвертація mp3 → wav
        audio = AudioSegment.from_mp3(temp_audio.name)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(temp_wav.name, format="wav")

    with st.spinner("🧠 Визначаємо таймінги..."):
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(temp_wav.name, beam_size=5)
        lines = list(segments)

    with st.spinner("🖼️ Підбираємо зображення..."):
        # Отримання зображень з Pexels
        headers = {"Authorization": PEXELS_API_KEY}
        img_clips = []

        for segment in lines:
            query = segment.text.split()[0] if segment.text else "nature"
            params = {"query": query, "per_page": 1}
            response = requests.get(pexels_url, headers=headers, params=params)
            data = response.json()
            try:
                img_url = data["photos"][0]["src"]["landscape"]
            except:
                img_url = "https://via.placeholder.com/1280x720.png?text=No+Image"

            img_data = requests.get(img_url).content
            img = Image.open(BytesIO(img_data)).resize((1280, 720))
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
    st.info("⬆️ Введіть текст сценарію і натисніть кнопку.")
