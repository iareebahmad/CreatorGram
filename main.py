import streamlit as st
from pydub import AudioSegment
import tempfile
import os
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="CreatorGram", layout="centered")
st.title("CreatorGram")
st.markdown("Upload a short video (or audio). The app will transcribe the first 15s and generate hook lines + thumbnail text suggestions.")

# Sidebar instructions
st.sidebar.header("Instructions")
st.sidebar.markdown(
    """
- Upload MP4 / MOV / MP3 / WAV (keep it short — ~15 seconds recommended).
- If transcription fails, paste a short transcript in the box.
- This is an MVP: for production use host STT & LLM inference on dedicated servers.
"""
)

# File uploader
uploaded_file = st.file_uploader("Upload video or audio file", type=["mp4","mov","mkv","mp3","wav","m4a"])

# Optional: user's niche / audience to tailor the hooks
audience = st.text_input("Target audience / niche (optional)", placeholder="eg: travel lovers, fitness beginners, tech founders")

# Load LLM (FLAN-T5 small) lazily
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

# Helper: convert video to wav (first 15s)
def extract_audio_segment_to_wav(input_bytes, start_ms=0, duration_ms=15000):
    # write bytes to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix="") as tf:
        tf.write(input_bytes)
        tmp_path = tf.name
    # pydub can open many container formats if ffmpeg is present
    try:
        audio = AudioSegment.from_file(tmp_path)
        # slice
        end_ms = min(len(audio), start_ms + duration_ms)
        clip = audio[start_ms:end_ms]
        out_path = tmp_path + ".wav"
        clip.export(out_path, format="wav")
        return out_path
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# Helper: transcribe wav using SpeechRecognition + Google Web Speech
def transcribe_wav_google(wav_path):
    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = r.record(source)
    try:
        # Uses Google Web Speech API (internet required). Short clips work best.
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        return None

# Prompt template for hook generation
HOOK_PROMPT_TEMPLATE = (
    "You are a creative assistant that writes short, attention-grabbing opening hooks for short-form videos. "
    "Given the transcript of the first few seconds of the video and the target audience, produce 5 short hook lines (each <= 18 words), and 5 thumbnail text suggestions (each <= 4 words). "
    "Make hooks urgent, curiosity-driven, or benefit-led. Return JSON-like sections labeled HOOKS and THUMBS.\n\n"
    "Transcript: {transcript}\n"
    "Audience: {audience}\n"
)

# LLM inference function
def generate_hooks_and_thumbs(transcript, audience, tokenizer, model, max_output_tokens=128):
    prompt = HOOK_PROMPT_TEMPLATE.format(transcript=transcript, audience=(audience or "general audience"))
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens= max_output_tokens, do_sample=True, top_k=50, top_p=0.95, temperature=0.9)
    gen = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return gen

# Main pipeline
if uploaded_file is not None:
    st.info("Processing — extracting audio and transcribing the first 15 seconds...")
    bytes_data = uploaded_file.read()
    try:
        wav_path = extract_audio_segment_to_wav(bytes_data, start_ms=0, duration_ms=15000)
    except Exception as e:
        st.error(f"Failed to extract audio: {e}")
        wav_path = None

    transcript = None
    if wav_path:
        with st.spinner("Transcribing (Google Web Speech)..."):
            transcript = transcribe_wav_google(wav_path)
        if transcript:
            st.success("Transcription complete — here's the detected text:")
            st.info(transcript)
        else:
            st.warning("Automatic transcription failed or was unreliable. Please paste a short transcript below.")

    # Let user edit or paste transcript
    transcript_input = st.text_area(
        "Transcript (edit or paste if auto-transcription failed)",
        value=(transcript or ""),
        height=120
    )

    # Generate hooks + thumbnail ideas
    if st.button("Generate Hooks & Thumbnails"):
        if not transcript_input.strip():
            st.error("Please provide a short transcript.")
        else:
            with st.spinner("Loading model and generating ideas..."):
                tokenizer, model = load_model()
                gen = generate_hooks_and_thumbs(transcript_input.strip(), audience.strip(), tokenizer, model)

            st.subheader("📌 Hooks")
            hooks_section = [line.strip() for line in gen.splitlines() if line.strip()][:5]
            for i, hook in enumerate(hooks_section, start=1):
                st.write(f"{i}. {hook}")

            st.subheader("🖼 Thumbnail Text Ideas")
            thumbs_section = [line.strip() for line in gen.splitlines() if len(line.split()) <= 4][:5]
            for thumb in thumbs_section:
                st.write(f"- {thumb}")

else:
    st.info("Upload a short video or audio to begin. You can also paste a short transcript directly.")


# Footer / notes
st.markdown("---")
st.markdown(
        '<p style="text-align:center;">Made with ❤️ by Areeb</p>',
        unsafe_allow_html=True
    )