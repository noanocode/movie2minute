import os
import streamlit as st
import tempfile
import whisper
import openai
from dotenv import load_dotenv
import torch
import torchaudio
import pandas as pd
from moviepy.editor import VideoFileClip
from pyannote.audio import Pipeline

# 環境変数をロード
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pyannoteパイプラインの初期化
hf_token = os.getenv("HUGGINGFACE_TOKEN")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
pipeline.params = {"min_duration_on": 0.5}  # パラメータをここで設定

# 音声抽出
def extract_audio_from_video(video_path, output_audio_path):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio_path)
        video.close()
    except Exception as e:
        st.error(f"Audio extraction failed: {str(e)}")

# 話者分離
def perform_speaker_diarization(audio_path):
    try:
        diarization = pipeline(audio_path)
        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return results
    except Exception as e:
        st.error(f"Diarization error: {str(e)}")
        return None

# 音声の文字起こし
def transcribe_audio(audio_path):
    try:
        model = whisper.load_model("large")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

# 各発言に話者ラベルを付ける
def label_sentences_with_speakers(transcription, diarization_results):
    sentences = transcription.split('.')
    labeled_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_start = transcription.index(sentence)
        sentence_end = sentence_start + len(sentence)
        sentence_midpoint = (sentence_start + sentence_end) / 2
        
        closest_segment = min(diarization_results, 
                              key=lambda x: abs((x['start'] + x['end'])/2 - sentence_midpoint))
        
        start_time = pd.to_datetime(closest_segment['start'], unit='s').strftime('%H:%M:%S')
        
        labeled_sentences.append({
            'speaker': closest_segment['speaker'],
            'text': sentence,
            'start_time': start_time  # 開始時刻の追加
        })
    
    return labeled_sentences

# Streamlit UI
st.title("動画から話者分離と文字起こし")

# 動画のアップロード
uploaded_file = st.file_uploader("動画ファイルをアップロードしてください (mp4形式)", type=["mp4"])

if uploaded_file:
    # 一時ファイルへの保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_file.write(uploaded_file.read())
        temp_video_path = temp_video_file.name

    # 音声抽出
    audio_path = tempfile.mktemp(suffix=".wav")
    extract_audio_from_video(temp_video_path, audio_path)

    # 議事録作成開始
    if st.button("議事録作成開始"):
        # 文字起こし
        with st.spinner("文字起こし中..."):
            transcription = transcribe_audio(audio_path)

            if transcription:
                st.subheader("文字起こし結果")
                st.text_area("全文", transcription, height=300)

                # 話者分離
                with st.spinner("話者分離を実行中..."):
                    diarization_results = perform_speaker_diarization(audio_path)

                    if diarization_results:
                        # ラベル付け
                        labeled_sentences = label_sentences_with_speakers(transcription, diarization_results)
                        
                        # 話者分離結果の表示
                        st.subheader("話者分離結果")
                        for sentence in labeled_sentences:
                            st.write(f"{sentence['speaker']} ({sentence['start_time']}): {sentence['text']}")

                        # 議事録の表示
                        st.subheader("議事録（スピーカーと内容）")
                        df = pd.DataFrame(labeled_sentences)
                        st.dataframe(df, use_container_width=True)

                        # CSVダウンロード
                        csv = df.to_csv(index=False)
                        st.download_button("CSVでダウンロード", data=csv, file_name="議事録.csv", mime="text/csv")