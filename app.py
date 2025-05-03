import streamlit as st
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from faster_whisper import WhisperModel
from pydub import AudioSegment
import requests
from io import BytesIO
from PIL import Image
import tempfile
import edge_tts
import asyncio

# --- Константи ---
PEXELS_API_KEY = st.secrets["PEXELS_API_KEY"]
pexels_url = "https://api.pexels.com/v1/search"

st.set_page_config(page_title="🎬 Відеогенератор", layout="centered")
st.title("🎬 Автоматичне відео з озвученням")

# --- Ввід сценарію ---
script = st.text_area("Введіть текст сценарію", height=200)

# --- Edge TTS синтез ---
async def generate_tts(text, output_path):
    communicate = edge_tts.Communicate(text, voice="uk-UA-OstapNeural")
    await communicate.save(output_path)

# --- Обробка кнопки ---
if st.button("🎥 Згенерувати відео") and script.strip() != "":
    with st.spinner("🔊 Генеруємо озвучку (EdgeTTS)..."):
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        asyncio.run(generate_tts(script, audio_path))

        audio = AudioSegment.from_mp3(audio_path)
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
            keywords = segment.text.split()[:3]
            query = " ".join(keywords) if keywords else "abstract"
            params = {"query": query, "per_page": 1}
            try:
                response = requests.get(pexels_url, headers=headers, params=params, timeout=5)
                data = response.json()
                img_url = data["photos"][0]["src"]["landscape"]
            except:
                img_url = None

            try:
                if img_url:
                    img_data = requests.get(img_url, timeout=5).content
                    img = Image.open(BytesIO(img_data)).resize((1280, 720))
                else:
                    img = Image.new('RGB', (1280, 720), color=(0, 0, 0))
            except:
                img = Image.new('RGB', (1280, 720), color=(0, 0, 0))

            duration = segment.end - segment.start
            clip = ImageClip(img).set_duration(duration)
            img_clips.append(clip)

    with st.spinner("🎞️ Монтуємо відео..."):
        final_video = concatenate_videoclips(img_clips, method="compose")
        final_video = final_video.set_audio(AudioFileClip(audio_path))

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        final_video.write_videofile(output_path, fps=24)

    st.success("✅ Відео згенеровано!")
    st.video(output_path, use_container_width=True)

else:
    st.info("⬆️ Введіть текст сценарію і натисніть кнопку.")
