import streamlit as st
import openai
import wave
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
from moviepy.editor import VideoFileClip, AudioFileClip
import os
import requests
from pydub import AudioSegment
from dotenv import load_dotenv
import json

load_dotenv()
credential_path = os.getenv('GOOGLE_CREDENTIALS_FILE')
with open(credential_path, 'w') as creds_file:
    creds_file.write(os.getenv('CREDS'))

def convert_to_mono(audio_path):
    sound = AudioSegment.from_file(audio_path)
    sound = sound.set_channels(1) 
    mono_audio_path = "mono_audio.wav"
    sound.export(mono_audio_path, format="wav")
    return mono_audio_path

def transcribe_audio(audio_path):
    mono_audio_path = convert_to_mono(audio_path)
    with wave.open(mono_audio_path, "rb") as audio_file:
        sample_rate = audio_file.getframerate()

    client = speech.SpeechClient.from_service_account_file(credential_path)

    with open(mono_audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="en-US",
        enable_automatic_punctuation=True
    )

    response = client.recognize(config=config, audio=audio)

    transcription = ' '.join([result.alternatives[0].transcript for result in response.results])
    return transcription



def generate_audio(text):
    if not text.strip():
        raise ValueError("The input text for Text-to-Speech is empty. Please provide valid text.")

    client = texttospeech.TextToSpeechClient.from_service_account_file(credential_path)

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-J",  
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    output_audio_path = "generated_audio.wav"
    with open(output_audio_path, "wb") as out:
        out.write(response.audio_content)

    print(f"Audio content written to file: {output_audio_path}")
    return output_audio_path

def adjust_audio_length(audio_path, video_path):
    video = VideoFileClip(video_path)
    video_duration = video.duration * 1000 

    audio = AudioSegment.from_file(audio_path)
    adjusted_audio = audio

    if len(audio) < video_duration:
        silence_duration = video_duration - len(audio)
        silence = AudioSegment.silent(duration=silence_duration)
        adjusted_audio = audio + silence
    else:
        adjusted_audio = audio[:video_duration]

    adjusted_audio_path = "adjusted_audio.wav"
    adjusted_audio.export(adjusted_audio_path, format="wav")
    return adjusted_audio_path

def replace_audio_in_video(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(audio_path)
    video = video.set_audio(new_audio)
    video.write_videofile(output_path)


def transcribe_audio_with_word_timestamps(audio_path):
    mono_audio_path = convert_to_mono(audio_path)
    
    client = speech.SpeechClient.from_service_account_file(credential_path)

    with open(mono_audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True  # Capture word-level timestamps
    )

    response = client.recognize(config=config, audio=audio)
    
    sentences = []
    current_sentence = {"transcript": "", "start_time": None, "end_time": None}

    for result in response.results:
        for word_info in result.alternatives[0].words:
            word = word_info.word
            start_time = word_info.start_time.total_seconds()
            end_time = word_info.end_time.total_seconds()

            if current_sentence["start_time"] is None:
                current_sentence["start_time"] = start_time

            current_sentence["transcript"] += word + " "
            current_sentence["end_time"] = end_time

            if word.endswith(('.', '?', '!')):
                sentences.append(current_sentence)
                current_sentence = {"transcript": "", "start_time": None, "end_time": None}

    if current_sentence["transcript"]:  
        sentences.append(current_sentence)

    return sentences

def correct_transcription(transcription_text):
    azure_openai_key = os.getenv('OPENAI_KEY')
    azure_openai_endpoint = os.getenv('OPENAI_ENDPOINT')

    if not transcription_text:
        st.error("Transcription is empty or None.")
        return None

    headers = {
        "Content-Type": "application/json",
        "api-key": azure_openai_key
    }
    data = {
        "messages": [
            {"role": "system", "content": "You are an assistant that corrects grammar and removes filler words."},
            {"role": "user", "content": transcription_text}
        ]
    }

    response = requests.post(azure_openai_endpoint, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        corrected_text = result["choices"][0]["message"]["content"].strip()
        return corrected_text
    else:
        st.error(f"Failed to retrieve response: {response.status_code} - {response.text}")
        return None

def generate_audio_with_dynamic_ssml(corrected_sentences, original_sentences):
    client = texttospeech.TextToSpeechClient.from_service_account_file(credential_path)

    ssml = "<speak>"

    for i, sentence in enumerate(corrected_sentences):
        if i < len(original_sentences):
            original_sentence = original_sentences[i]
            start_time = original_sentence['start_time']
            end_time = original_sentence['end_time']
            sentence_duration = end_time - start_time

            ssml += f'<break time="{start_time}s"/> {sentence.strip()}. <break time="{sentence_duration}s"/>'

    ssml += "</speak>"

    synthesis_input = texttospeech.SynthesisInput(ssml=ssml)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-J",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    output_audio_path = "generated_audio_dynamic_ssml.wav"
    with open(output_audio_path, "wb") as out:
        out.write(response.audio_content)

    return output_audio_path

def replace_audio_in_video(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(audio_path)
    video = video.set_audio(new_audio)
    video.write_videofile(output_path)

st.title("AI Audio Correction with Dynamic Sentence Alignment")
st.write("Upload a video file, and we'll replace the audio with an AI-generated corrected voice, dynamically matching the timing.")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

if uploaded_file:
    video_path = uploaded_file.name
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Processing the video...")

    video_clip = VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    video_clip.audio.write_audiofile(audio_path)

    transcription_data = transcribe_audio_with_word_timestamps(audio_path)
    original_transcription = " ".join([item["transcript"] for item in transcription_data])
    st.write("Original Transcription:", original_transcription)

    corrected_transcription = correct_transcription(original_transcription)
    st.write("Corrected Transcription:", corrected_transcription)

    corrected_sentences = corrected_transcription.split('. ')

    new_audio_path = generate_audio_with_dynamic_ssml(corrected_sentences, transcription_data)

    output_video_path = "output_video_dynamic.mp4"
    replace_audio_in_video(video_path, new_audio_path, output_video_path)

    st.write("Here is the output video with the corrected and dynamically synchronized audio:")
    st.video(output_video_path)
