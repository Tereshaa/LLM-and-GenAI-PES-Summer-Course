import whisper
import streamlit as st
import openai
import base64

openai.api_key = 'api_key_here'

st.set_page_config(page_title="Audio to Text Transcription and Translation")
st.header("Audio to Text Transcription and Translation")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3"])

def save_as_txt(text, file_name):
    try:
        with open(f"{file_name}.txt", "w") as f:
            f.write(text)
            print("\nThe text saved as a text file with name", file_name)
    except Exception as e:
        print("\nThe file can't be saved as a text file:", str(e))

def translate_text(text, target_language="en"):
    # Translate using OpenAI GPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Translate the following text into {target_language}:\n\n{text}"}
        ]
    )

    translation = response['choices'][0]['message']['content'].strip()
    print("The translation:")
    print(translation)
    return translation

def use_audio(file_path):
    whisper_model = whisper.load_model("base")
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)
    log_mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    _, probs = whisper_model.detect_language(log_mel)
    detected_language = max(probs, key=probs.get)
    print("The audio is in:", detected_language, "language")
    
    # Transcribe audio in the detected language
    decoding_options = whisper.DecodingOptions(language=detected_language)
    decoded_result = whisper.decode(whisper_model, log_mel, decoding_options)
    text = decoded_result.text
    print("\nThe text extracted from the audio is:\n", text)
    
    # Save transcription as text file
    file_name = file_path.split(".")[0].split("/")[-1]  # Extracting file name without extension
    save_as_txt(text, file_name)
    
    translation = None
    if detected_language != 'en':
        translation = translate_text(text, "en")  # Translate to English
    
    return detected_language, text, translation, file_name

def main_func():
    if uploaded_file is not None:
        file_name = uploaded_file.name.split(".")[0]
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        detected_language, text, translation, file_name = use_audio(file_path)
        return detected_language, text, translation, file_name

submit = st.button("Submit")

if submit:
    detected_language, text, translation, file_name = main_func()
    if text:
        st.subheader("Detected Language:\n")
        st.write(detected_language)
        st.subheader("Transcription:\n")
        st.write(text)

        # Downloadable links for transcription
        text_filename = f"{file_name}_transcription.txt"
        text_b64 = base64.b64encode(text.encode()).decode()
        href = f'<a href="data:text/plain;base64,{text_b64}" download="{text_filename}">Download Transcription</a>'
        st.markdown(href, unsafe_allow_html=True)

        if translation:
            st.subheader("Translation to English:\n")
            st.write(translation)

            # Downloadable links for translation
            translation_filename = f"{file_name}_translation.txt"
            translation_b64 = base64.b64encode(translation.encode()).decode()
            href_trans = f'<a href="data:text/plain;base64,{translation_b64}" download="{translation_filename}">Download Translation</a>'
            st.markdown(href_trans, unsafe_allow_html=True)
