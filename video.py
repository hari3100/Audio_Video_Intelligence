from datetime import datetime
import streamlit as st
import json
import os
import pandas as pd
from video_functions  import *
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


st.set_page_config(page_title="Incrify Audio Video Intelligence",
                   page_icon="🗣️",
                   layout="wide")


INPUT_FOLDER = ""
OUTPUT_FOLDER = "Output_files"
aai_api_key = st.secrets["aai_api_key"]

# ======================================================================================
# ================================ UTILITIES ===========================================
# ======================================================================================

def custom_normalize_path(path):
    # Replace all escape sequences with their raw string representations
    replacements = {'\\': r'\\', '\a': r'\a', '\b': r'\b', '\f': r'\f', '\n': r'\n', '\r': r'\r', '\t': r'\t', '\v': r'\v'}
    for old, new in replacements.items():
        path = path.replace(old, new)
    # Normalize the path using os.path.normpath
    normalized_path = os.path.normpath(path)
    return normalized_path

def save_uploaded_file(uploaded_file):
    try:
        # Create a directory to save the file
        save_path = 'uploaded_files'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Create a unique filename to prevent file overwrites
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{current_time}_{uploaded_file.name}"
        complete_path = os.path.join(save_path, file_name)

        # Write the file to the directory
        with open(complete_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return complete_path
    except Exception as e:
        return str(e)

# ======================================================================================
# ================================ UTILITIES ===========================================
# ======================================================================================

def VI(filepath):
    # filepath = os.path.join(INPUT_FOLDER, filepath)
    try:
        if filepath.split(".")[-1] == 'mp4':
            # getting video emotions
            logging.info(f"Stage 1 Video Emotions analyzer : Initiated")
            video_emotions = process_video(filepath)
            logging.info(f"Stage 1 Video Emotions analyzer : Completed Succesfully")

            # # segregating audios from videos
            logging.info(f"Stage 1.1 segregating audios from videos : Initiated")
            filepath = segregate_audio_from_video(filepath)
            logging.info(f"Stage 1.1 segregating audios from videos : Completed Succesfully")
            logging.info(f"Extracted Audio file from Video saved at {filepath}")
          
            return video_emotions, filepath
        else:
            video_emotions = {"video_emotions" : "No video Found"}
            return video_emotions, filepath
    except Exception as e:
        st.json(({"error": str(e)}))

def AI(filepath):
    try:
        # stage 2
        # st.write(f"performing Audio Intellegence on : {filepath}")
        logging.info(f"Stage 2 Transcribing Audio File located at {filepath} : Initiated")
        transcript, emotion_dictionary, time_stamps = Transcribe_Text(filepath, aai_api_key)
        logging.info(f"Stage 2 Transcribing Audio File : Completed Succesfully")

        # stage 3
        logging.info(f"Stage 3 - Segregating & Exporting Speakers Audio : Initiated")
        no_of_speaker, total_speakers_urls, blank_dict = Export_Speakers_Audio(filepath, OUTPUT_FOLDER, time_stamps)
        # st.success(f"no_of_speaker: {no_of_speaker}")
        # st.success(f"total_speakers_urls: {total_speakers_urls}")
        logging.info(f"Stage 3 - Segregating & Exporting Speakers Audio : Completed Succesfully")
        for i in total_speakers_urls:
            logging.info(f"Expoert audio of speaker '{i}' at '{total_speakers_urls[i]}' ")


  

        # stage 4
        logging.info(f"Stage 4 - Speaker Analysis : Initiated")
        for speaker, speaker_url in total_speakers_urls.items():
            blank_dict[speaker] = Speaker_Analsis(speaker_url, aai_api_key)
        logging.info(f"Stage 4 - Speaker Analysis : Completed Succesfully")

        # Return the result as JSON
        return {"message": "File processed successfully",
                "transcribed_text": transcript,
                "speaker_emotions": emotion_dictionary,
                "total_speaker": no_of_speaker,
                "total_speakers_urls": total_speakers_urls,
                "speaker_analsis": blank_dict }


    except Exception as e:
        st.json({"error": str(e)})




# Streamlit app
def main():

    # External links
    st.sidebar.title("Our Other Projects")
    st.sidebar.markdown("<a href='https://incrify90.streamlit.app/' target='_blank'>Chat with PDF's</a>", unsafe_allow_html=True)
    st.sidebar.markdown("<a href='https://incrify-avi.streamlit.app/' target='_blank'>Audio Video Intelligence</a>", unsafe_allow_html=True)
    st.sidebar.markdown("<a href='https://incrify-ai-course-builder.streamlit.app/' target='_blank'>Ai Course Builder</a>", unsafe_allow_html=True)

  
    st.title("Audio - Video Intelligence")

    # File uploader
    uploaded_file = st.file_uploader("Upload an audio or video file", type=["mp4","wav"])

    # Process button
    if st.button("Process"):
        if uploaded_file is not None:
            file_path = save_uploaded_file(uploaded_file)
            logging.info(f"File saved at: {file_path}")

            logging.info(f"Performing Video Analysis")
            video_emotions, extracted_mp3_filepath = VI(file_path)
            logging.info(f"Video Analysis Completed")

            st.title('Video Analsis')
            # st.write(output['video_emotions'])
            for metrics, value in (video_emotions.items()):
                st.markdown(f"{metrics}: {value}")
            st.markdown("---")
            # st.write(video_emotions)
          
            logging.info(f"Performing Audio Analysis")
            output = AI(extracted_mp3_filepath)           
            logging.info(f"Video Analysis Completed")

            # st.write(output)
            
            st.title(f"Total no.of speakers : {output['total_speaker']}")
            st.markdown("---")

            st.title("Transcribed Text and Emotions")
            transcribed_data = display_transcribed_text_with_emotions(output)
            st.table(transcribed_data)
            st.markdown("---")

            st.title('Speaker Analsis')
            cols = st.columns(output['total_speaker'])  # Create a column for each speaker

            for i, (speaker, metrics) in enumerate(output['speaker_analsis'].items()):
                with cols[i]:
                    st.markdown(f"**Speaker {speaker}:**")
                    for metric, value in metrics.items():
                        st.markdown(f"{metric}: {value}")
            st.markdown("---")

            


        else:
            st.error("Please upload a file first.")

if __name__ == "__main__":
    main()
