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

load_dotenv()
# Convert stereo to mono using pydub
credential_path = os.getenv('GOOGLE_CREDENTIALS_FILE')

def convert_to_mono(audio_path):
    sound = AudioSegment.from_file(audio_path)
    sound = sound.set_channels(1)  # Set to mono
    mono_audio_path = "mono_audio.wav"
    sound.export(mono_audio_path, format="wav")
    return mono_audio_path

# Transcribe audio using Google Speech-to-Text API
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

# Correct transcription using GPT-4


# Generate audio from the corrected transcription
def generate_audio(text):
    if not text.strip():
        raise ValueError("The input text for Text-to-Speech is empty. Please provide valid text.")

    client = texttospeech.TextToSpeechClient.from_service_account_file(credential_path)

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-J",  # Example: Journey voice model
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

# Adjust audio length to match video length
def adjust_audio_length(audio_path, video_path):
    video = VideoFileClip(video_path)
    video_duration = video.duration * 1000  # in milliseconds

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

# Replace the audio in the video with the generated one
def replace_audio_in_video(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(audio_path)
    video = video.set_audio(new_audio)
    video.write_videofile(output_path)



# Step 1: Transcribe audio and capture timestamps
import streamlit as st
import wave
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
from moviepy.editor import VideoFileClip, AudioFileClip
import requests
from pydub import AudioSegment


# Step 1: Transcribe audio with word-level timestamps and group them into sentences
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

            # Detect sentence end by punctuation
            if word.endswith(('.', '?', '!')):
                sentences.append(current_sentence)
                current_sentence = {"transcript": "", "start_time": None, "end_time": None}

    if current_sentence["transcript"]:  # Add any remaining sentence
        sentences.append(current_sentence)

    return sentences

# Step 2: Correct transcription using GPT-4
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

# Step 3: Generate audio using SSML with dynamically calculated pauses
def generate_audio_with_dynamic_ssml(corrected_sentences, original_sentences):
    client = texttospeech.TextToSpeechClient.from_service_account_file(credential_path)

    ssml = "<speak>"

    # Match corrected sentences with original sentence timings
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

# Step 4: Replace audio in the video
def replace_audio_in_video(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(audio_path)
    video = video.set_audio(new_audio)
    video.write_videofile(output_path)

# Step 5: Streamlit UI
st.title("AI Audio Correction with Dynamic Sentence Alignment")
st.write("Upload a video file, and we'll replace the audio with an AI-generated corrected voice, dynamically matching the timing.")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

if uploaded_file:
    video_path = uploaded_file.name
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Processing the video...")

    # Extract audio from video
    video_clip = VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    video_clip.audio.write_audiofile(audio_path)

    # Step 1: Transcribe the audio and get word-level timestamps grouped into sentences
    transcription_data = transcribe_audio_with_word_timestamps(audio_path)
    original_transcription = " ".join([item["transcript"] for item in transcription_data])
    st.write("Original Transcription:", original_transcription)

    # Step 2: Correct the transcription using GPT-4
    corrected_transcription = correct_transcription(original_transcription)
    st.write("Corrected Transcription:", corrected_transcription)

    corrected_sentences = corrected_transcription.split('. ')

    # Step 3: Generate SSML-based audio with dynamic sentence-level pauses
    new_audio_path = generate_audio_with_dynamic_ssml(corrected_sentences, transcription_data)

    # Step 4: Replace the original audio with the new one
    output_video_path = "output_video_dynamic.mp4"
    replace_audio_in_video(video_path, new_audio_path, output_video_path)

    st.write("Here is the output video with the corrected and dynamically synchronized audio:")
    st.video(output_video_path)
