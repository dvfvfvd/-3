import streamlit as st
from moviepy.editor import *
from pydub import AudioSegment
from gtts import gTTS
import requests
from PIL import Image
from io import BytesIO
import os
import tempfile

PEXELS_API_KEY = st.secrets["PEXELS_API_KEY"]

def search_image(query):
    headers = {
        "Authorization": PEXELS_API_KEY
    }
    response = requests.get(
        f"https://api.pexels.com/v1/search?query={query}&per_page=1", headers=headers)
    data = response.json()
    if data['photos']:
        return data['photos'][0]['src']['landscape']
    return None

def generate_tts(text, path):
    tts = gTTS(text)
    tts.save(path)

def create_video_script(script_text):
    scenes = [line.strip() for line in script_text.split('.') if line.strip()]
    clips = []
    temp_dir = tempfile.mkdtemp()

    for i, line in enumerate(scenes):
        st.write(f"🎙️ Генерується сцена {i+1}: {line}")
        image_url = search_image(line)
        if not image_url:
            continue
        img_response = requests.get(image_url)
        img = Image.open(BytesIO(img_response.content)).resize((1280, 720))
        img_path = os.path.join(temp_dir, f"scene_{i}.png")
        img.save(img_path)

        audio_path = os.path.join(temp_dir, f"scene_{i}.mp3")
        generate_tts(line, audio_path)
        audio = AudioFileClip(audio_path)
        duration = audio.duration

        clip = ImageClip(img_path).set_duration(duration).set_audio(audio).fadein(0.5).fadeout(0.5)
        clips.append(clip)

    final = concatenate_videoclips(clips)
    output_path = os.path.join(temp_dir, "final_video.mp4")
    final.write_videofile(output_path, fps=24)

    return output_path

st.title("🎬 Автоматичне створення відео зі сценарію")
script = st.text_area("Встав свій сценарій (розділяй репліки крапками)", height=200)

if st.button("🎥 Створити відео"):
    if not script.strip():
        st.warning("Будь ласка, введи сценарій.")
    else:
        with st.spinner("Обробка..."):
            video_path = create_video_script(script)
            st.success("✅ Відео створено!")
            st.video(video_path)
