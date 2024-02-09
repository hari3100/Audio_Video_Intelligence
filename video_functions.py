from datetime import datetime
import cv2
from PIL import Image
import numpy as np
import requests
from moviepy.editor import VideoFileClip
from transformers import pipeline
import os
import re
import time
from pydub import AudioSegment
import assemblyai as aai
import pandas as pd
import streamlit as st










# ======================================================================================
# ================================ VIDEO ===============================================
# ======================================================================================

facial_emotion_analyzer = pipeline("image-classification", model="dima806/face_emotions_image_detection")

def analyze_frames(frame):
    try:
        #         pil_image = Image.fromarray(frame)
        return facial_emotion_analyzer(Image.fromarray(frame))[0]
    except ValueError as e:
        print(f"Error: {e}")
        return None


def process_video(video_path, skip_frame=100):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return []

    emotions_timeline = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every (skip_frame + 1)th frame
        if frame_count % (skip_frame + 1) == 0:
            result = analyze_frames(frame)
            emotions_timeline.append(result)

        frame_count += 1

    # Release the video capture object
    cap.release()

    emotions_dictionary_default = {'Angry': 0, 'Ahegao': 0, 'Happy': 0, 'Sad': 0, 'Neutral': 0}
    video_emotions = emotions_dictionary_default.copy()
    for emotion in emotions_timeline:
        if emotion:
            emotions_dictionary_default[emotion["label"]] += 1

        #     print("=" * 50)
        if len(emotions_timeline) > 0:
            for every_emotional_class in emotions_dictionary_default:
                #             print(f"{every_emotional_class} ---> {(emotions_dictionary_default[every_emotional_class] / len(emotions_timeline)) * 100}")
                video_emotions[every_emotional_class] = (emotions_dictionary_default[every_emotional_class] / len(
                    emotions_timeline)) * 100

        else:
            print("No frames analyzed. emotions_timeline is empty.")
            video_emotions = 0

    return video_emotions


# ======================================================================================
# ================================ AUDIO ===============================================
# ======================================================================================

# stage 1
# takes video and extracts audio from it and return audio.wav
def segregate_audio_from_video(input_file):
    
    # Getting file name
    # file_name = input_file.rsplit('.', 1)[0].rsplit('\\', 1)[1]
    file_name = (input_file.rsplit('.')[0]).split(r"/")[1]

    # Load the video clip
    video_clip = VideoFileClip(input_file)

    # Separate audio
    audio_clip = video_clip.audio

    save_path = os.path.dirname(input_file)

    # Save audio to wav file (Google Cloud Speech-to-Text supports LINEAR16 encoding)
    complete_path = os.path.join(save_path, f"extracted_audio_{file_name}.wav")

    # output = f"{output_location}\\extracted_audio_{file_name}.wav"
    audio_clip.write_audiofile(complete_path, codec='pcm_s16le', fps=16000)
    # st.write(f"exracted audio saved at : {complete_path}")
    return complete_path


# stage 2
# transcribing text
def Transcribe_Text(FILE_URL,aai_api_key):
    # t1 = time.time()
    # Replace with your API key
    aai.settings.api_key = aai_api_key

    config = aai.TranscriptionConfig(speaker_labels=True)

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(
      FILE_URL,
      config=config
    )
    # t2 =time.time()
    # print("Time taken",t2-t1)
    # print("*"*50)
    # print("Transcribed text with speakers bifurcations")
    # print("*"*50)

    # used too capture the sentiments of the speakers for the whole text
    emotion_dictionary = {}

    return_transcribe = []
    for utterance in transcript.utterances:
        return_transcribe.append({utterance.speaker : utterance.text})
        # print(f"{(utterance.start, utterance.end)} | Speaker {utterance.speaker}: {utterance.text}")
        # print(get_emotions(utterance.text))
        if utterance.speaker not in emotion_dictionary:
          emotion_dictionary[utterance.speaker]=[]

        emotion_dictionary[utterance.speaker].append(get_emotions(utterance.text)[0])

    time_stamps = [(utterance.start, utterance.end, utterance.speaker) for utterance in transcript.utterances]
    return return_transcribe, emotion_dictionary, time_stamps
    # print()
    # print("*"*50)

    # print("Segeregating Speakers")
    # print("*"*50)
    # time_stamps =[(utterance.start, utterance.end, utterance.speaker) for utterance in transcript.utterances]
    # Export_Speakers_Audio(FILE_URL, time_stamps,aai_api_key)


# stage 3
# count speakers and seg audio  wrt speakeres
def Export_Speakers_Audio(FILE_URL, speakers_output_location, time_stamps):
    # Load your master audio file
    master_audio = AudioSegment.from_file(FILE_URL)

    # Initialize segments for each speaker
    segments = {speaker: AudioSegment.silent(duration=0) for _, _, speaker in time_stamps}



    no_of_speaker = len(segments.keys())
    # print(f"Number of speakers identified : {no_of_speaker}")

    # Extract segments for each speaker
    for start, end, speaker in time_stamps:
        segment = master_audio[start:end]
        segments[speaker] += segment

        # Export each speaker's audio to separate files
    total_speakers_urls = {speaker: [] for _, _, speaker in time_stamps}
    for speaker, audio in segments.items():
        audio.export(f"{speakers_output_location}/speaker_{speaker}.wav", format="wav")
        print(f"Exported speaker_{speaker}.wav")
        total_speakers_urls[speaker] = f"{speakers_output_location}/speaker_{speaker}.wav"
        # total_speakers_urls.append(f"{speakers_output_location}/speaker_{speaker}.wav")

    blank_dict = {speaker: [] for _, _, speaker in time_stamps}

    return no_of_speaker, total_speakers_urls, blank_dict

    # print()
    # print("*"*50)
    # print("Analyisizing speakers")
    # print("*"*50)
    # for i in segments.keys():
    #     print("-"*10)
    #     print(f"Summary of speaker_{i}")
    #     print("-"*10)
    #     Speaker_Analsis(f"speaker_{i}.wav",aai_api_key)


# stage 4
# speaker analysis individually (wpm, filler_words_per_minute, avg_pause_duration,longest_pause )
def Speaker_Analsis(SPEAKER_FILE_URL,aai_api_key):

    aai.settings.api_key = aai_api_key

    transcriber = aai.Transcriber()

    # Transcribe the file
    transcript = transcriber.transcribe(SPEAKER_FILE_URL)

    # Wait for the transcription to complete
    while transcript.status != 'completed':
        pass

    total_words = 0
    total_time = 0
    total_pauses = 0
    longest_pause = 0
    filler_words = 0
    last_end_time = 0

    for word in transcript.words:
        # Count total words
        total_words += 1

        # Check for filler words
        if word.text.lower() in ['um', 'uh', 'er', 'ah','umm','ahh']:
            filler_words += 1

        # Calculate pause durations
        pause_duration = word.start - last_end_time
        if pause_duration > 0:
            total_pauses += pause_duration
            longest_pause = max(longest_pause, pause_duration)

        last_end_time = word.end

    total_time = transcript.words[-1].end - transcript.words[0].start
    wpm = (total_words / total_time) * 60000
    filler_words_per_minute = (filler_words / total_time) * 60000
    avg_pause_duration = (total_pauses / (len(transcript.words) - 1) if total_words > 1 else 0) / 1000
    longest_pause = longest_pause / 1000

    return {
        'Words Per Minute': f'{wpm:.2f}',
        'Average Pause Duration (seconds)': f'{avg_pause_duration:.2f}',
        'Longest Pause (seconds)': f'{longest_pause:.2f}',
        'Filler Words Per Minute': f'{filler_words_per_minute:.2f}'
    }


emotion_pipeline = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
def get_emotions(text):
    # Emotion analysis using the pre-created pipeline
    emotion_labels = emotion_pipeline(text)
    return emotion_labels


def get_first_valid_emotion(speaker, emotions):
    for i, emotion in enumerate(emotions[speaker]):
        if "label" in emotion and "score" in emotion:
            emotions[speaker].pop(i)  # Remove the used emotion from the list
            return emotion["label"], emotion["score"]
    return "N/A", "N/A"


def display_transcribed_text_with_emotions(data):
    formatted_data = []
    for item in data["transcribed_text"]:
        if item:
            for speaker, text in item.items():
                emotion, score = get_first_valid_emotion(speaker, data["speaker_emotions"])
                formatted_data.append({"Speaker": speaker, "Text": text, "Emotion": emotion, "Score": score})

    return pd.DataFrame(formatted_data, columns=["Speaker", "Text", "Emotion", "Score"])








