import streamlit as st
import fitz  # PyMuPDF
import tempfile
import os
import numpy as np
import wave
from gtts import gTTS
import sounddevice as sd
from st_audiorec import st_audiorec
from huggingface_hub import InferenceClient
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ====== Language Dictionaries ======
LANGUAGES = {
    "ar": {
        "title": "المعلم الصغير",
        "welcome": "مرحباً بك في المعلم الصغير",
        "book_viewer": "عارض الكتاب",
        "choose_book": "اختر كتاباً للقراءة",
        "no_book": "لم يتم اختيار كتاب بعد",
        "prev": "السابق",
        "next": "التالي",
        "page": "صفحة: {}/{}",
        "send": "إرسال",
        "speak": "تحدث",
        "loading": "جاري التحميل...",
        "ready": "جاهز للدردشة",
        "recording": "جاري التسجيل...",
        "thinking": "جاري التفكير...",
        "speaking": "جاري النطق...",
        "error": "حدث خطأ: {}",
        "please_upload": "الرجاء تحميل ملف PDF أولاً",
        "teacher": "المعلم: ",
        "me": "أنا: ",
    },
    "en": {
        "title": "Little Teacher",
        "welcome": "Welcome to Little Teacher",
        "book_viewer": "Book Viewer",
        "choose_book": "Choose a Book",
        "no_book": "No book selected yet",
        "prev": "Previous",
        "next": "Next",
        "page": "Page: {}/{}",
        "send": "Send",
        "speak": "Speak",
        "loading": "Loading...",
        "ready": "Ready to chat",
        "recording": "Recording...",
        "thinking": "Thinking...",
        "speaking": "Speaking...",
        "error": "Error: {}",
        "please_upload": "Please upload a PDF file first",
        "teacher": "Teacher: ",
        "me": "Me: ",
    }
}

# ====== Initial Prompts ======
INITIAL_PROMPTS = {
    "ar": """
    اقرأ محتوى الملف وقم بإنشاء:
    1. تشويق صغير عن موضوع الملف (حقيقة مثيرة أو معلومة مدهشة)
    2. سؤال بسيط متعلق بالموضوع
    اجعل الأسلوب مناسباً للأطفال ومشوقاً
    """,
    "en": """
    Read the content and create:
    1. A fun and exciting teaser about the topic (an interesting fact or amazing information)
    2. A simple question about the topic
    Make it child-friendly and engaging using simple language
    """
}

# ====== Prompt Templates ======
conversation_template = PromptTemplate(
    input_variables=["history", "kid_answer"],
    template="""
أنت معلم أطفال ودود وحماسي، مهمتك هي:
1. في البداية، اطرح سؤالاً تحفيزياً بسيطاً أو العب لعبة التخمين مع الطفل حول موضوع الكتاب
2. بعد إجابة الطفل، اشرح محتوى الكتاب بطريقة مبسطة وممتعة
3. استخدم أسلوباً تفاعلياً وقصصياً في الشرح
4. اطرح أسئلة تفكير بسيطة بين الحين والآخر

المحادثة السابقة:
{history}

إجابة الطفل: {kid_answer}

المعلم يرد (يجب أن يتضمن الرد):
- تشجيع وتفاعل مع إجابة الطفل
- شرح جزء من المحتوى بأسلوب قصصي ممتع
- سؤال تفكير بسيط للمتابعة
"""
)

english_conversation_template = PromptTemplate(
    input_variables=["history", "kid_answer"],
    template="""
You are a friendly and enthusiastic teacher talking to a young child. Your task is to:
1. Be cheerful and encouraging, using simple words that children can understand
2. Break down complex ideas into simple, fun examples
3. Connect ideas to things children know and love (like toys, games, animals, or everyday experiences)
4. Keep sentences short and engaging

Previous Chat:
{history}

Child Said: {kid_answer}

Your Response Should:
1. Start with a greeting or praise
2. Explain things using fun examples and simple words
3. End with an easy and fun question

Your turn to respond:
"""
)

# ====== Helper Functions ======
def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name
    doc = fitz.open(tmp_path)
    text = "\n".join([page.get_text("text") for page in doc])
    page_count = doc.page_count
    doc.close()
    os.unlink(tmp_path)
    return text, page_count

def render_pdf_page(pdf_file, page_num):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name
    doc = fitz.open(tmp_path)
    page = doc[page_num]
    pix = page.get_pixmap()
    img_bytes = pix.tobytes("png")
    doc.close()
    os.unlink(tmp_path)
    return img_bytes

def text_to_speech(text, lang="ar"):
    try:
        tts = gTTS(text=text, lang=lang)
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_path = temp_audio.name
        temp_audio.close()
        tts.save(temp_audio_path)
        return temp_audio_path
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

def record_audio(duration=5, sample_rate=16000):
    st.info("🎤 Recording... Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_wav.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return temp_wav.name
def transcribe_audio(file_path, lang="ar"):
    api_key = "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"
    client = InferenceClient(api_key=api_key)
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    output = client.automatic_speech_recognition(
        audio_bytes,
        model="openai/whisper-large-v3"
    )
    return output.text.strip()

# ====== Streamlit App ======
st.set_page_config(page_title="Little Teacher", layout="wide")

# Sidebar: Language and PDF Upload
st.sidebar.title("Settings")
language = st.sidebar.radio("Language / اللغة", ["ar", "en"], format_func=lambda x: "العربية" if x == "ar" else "English")
texts = LANGUAGES[language]

uploaded_pdf = st.sidebar.file_uploader(texts["choose_book"], type="pdf", key="pdf_uploader")

# Session State Initialization
if "pdf_file" not in st.session_state:
    st.session_state.pdf_file = None
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "page_num" not in st.session_state:
    st.session_state.page_num = 0
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ai_ready" not in st.session_state:
    st.session_state.ai_ready = False

# PDF Handling
if uploaded_pdf:
    # Only rebuild vector DB and generate initial message if PDF is new
    if "pdf_file" not in st.session_state or st.session_state.pdf_file != uploaded_pdf:
        st.session_state.pdf_file = uploaded_pdf
        with st.spinner(texts["loading"]):
            text, total_pages = extract_text_from_pdf(uploaded_pdf)
            st.session_state.pdf_text = text
            st.session_state.total_pages = total_pages
            st.session_state.page_num = 0
            st.session_state.ai_ready = True

            # Build embeddings and FAISS index using AraTec.py logic
            from langchain.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain.chains import RetrievalQA
            from langchain.prompts import PromptTemplate

            # Chunk the PDF into paragraphs for vector storage
            chunks = [t.strip() for t in text.split('\n\n') if t.strip()]
            if not chunks:
                chunks = [text]
            try:
                # Build FAISS vector index from all chunks
                embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                faiss_index = FAISS.from_texts(chunks, embedding_model)
                retriever = faiss_index.as_retriever()

                # Set up LLM and RetrievalQA chain
                API_KEY = "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"
                llm = ChatOpenAI(
                    model="meta-llama/llama-4-scout:free",
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=API_KEY
                )
                from langchain.chains import RetrievalQA
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                st.session_state.qa_chain = qa_chain

                # Only generate the initial teacher message if chat_history is empty
                if not st.session_state.get("chat_history"):
                    initial_prompt = INITIAL_PROMPTS[language]
                    ai_response = qa_chain.run(initial_prompt)
                    st.session_state.chat_history = [{"role": "assistant", "content": ai_response}]

                    # Set a flag to trigger TTS for the initial message on the next rerun
                    st.session_state.initial_tts_played = False
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                ai_response = f"AI Error: {str(e)}\n{tb}"
                st.session_state.chat_history = [{"role": "assistant", "content": ai_response}]
    # Do not overwrite chat_history or rebuild vector DB on rerun

# Main Layout
st.title(texts["title"])
st.markdown(f"### {texts['welcome']}")

col1, col2 = st.columns([1, 1])

# PDF Viewer
with col1:
    st.subheader(texts["book_viewer"])
    if st.session_state.pdf_file:
        # Navigation
        prev, next_ = st.columns([1, 1])
        with prev:
            if st.button(texts["prev"], disabled=st.session_state.page_num == 0):
                st.session_state.page_num = max(0, st.session_state.page_num - 1)
        with next_:
            if st.button(texts["next"], disabled=st.session_state.page_num >= st.session_state.total_pages - 1):
                st.session_state.page_num = min(st.session_state.total_pages - 1, st.session_state.page_num + 1)
        # Display page
        st.markdown(texts["page"].format(st.session_state.page_num + 1, st.session_state.total_pages))
        # Rewind file pointer for rendering
        st.session_state.pdf_file.seek(0)
        img_bytes = render_pdf_page(st.session_state.pdf_file, st.session_state.page_num)
        st.image(img_bytes, use_column_width=True)
    else:
        st.info(texts["no_book"])

# Chat Interface
with col2:
    st.subheader("💬 " + texts["ready"])
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            st.markdown(f"**{texts['me']}** {entry['content']}")
        elif entry["role"] == "transcript":
            st.markdown(f"**{texts['me']} (من الصوت)** {entry['content']}")
        else:
            st.markdown(f"**{texts['teacher']}** {entry['content']}")
    # Play TTS for the initial message if not already played
    if (
        st.session_state.get("chat_history")
        and not st.session_state.get("initial_tts_played", False)
        and len(st.session_state.chat_history) == 1
        and st.session_state.chat_history[0]["role"] == "assistant"
    ):
        tts_path = text_to_speech(st.session_state.chat_history[0]["content"], lang=language)
        if tts_path:
            audio_file = open(tts_path, "rb")
            st.audio(audio_file.read(), format="audio/mp3")
            audio_file.close()
            os.remove(tts_path)
        st.session_state.initial_tts_played = True
    # User input
    # If a pending transcript exists, set it as the user_input before rendering the widget
    if "pending_transcript" in st.session_state:
        st.session_state["user_input"] = st.session_state["pending_transcript"]
        del st.session_state["pending_transcript"]
    user_input = st.text_input(texts["send"], key="user_input")
    if st.button(texts["send"]):
        if not st.session_state.pdf_file:
            st.warning(texts["please_upload"])
        elif user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            # AI response using RetrievalQA chain
            with st.spinner(texts["thinking"]):
                prompt_template = conversation_template if language == "ar" else english_conversation_template
                # Build chat history string for the prompt
                history = "\n".join(
                    f"{texts['me'] if m['role']=='user' else texts['teacher']}{m['content']}"
                    for m in st.session_state.chat_history if m["role"] != "assistant" or m != st.session_state.chat_history[-1]
                )
                qa_chain = st.session_state.get("qa_chain")
                if qa_chain:
                    try:
                        ai_response = qa_chain.run(
                            prompt_template.format(history=history, kid_answer=user_input)
                        )
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        ai_response = f"AI Error: {str(e)}\n{tb}"
                else:
                    ai_response = "AI Error: QA chain not initialized."
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            # Always render chat and play TTS for the latest assistant message
            if ai_response:
                st.markdown(f"**{texts['teacher']}** {ai_response}")
                tts_path = text_to_speech(ai_response, lang=language)
                if tts_path:
                    audio_file = open(tts_path, "rb")
                    st.audio(audio_file.read(), format="audio/mp3")
                    audio_file.close()
                    os.remove(tts_path)
    # TTS
    if st.session_state.chat_history and st.button(texts["speak"]):
        last_ai = next((entry for entry in reversed(st.session_state.chat_history) if entry["role"] == "assistant"), None)
        if last_ai:
            with st.spinner(texts["speaking"]):
                audio_path = text_to_speech(last_ai["content"], lang=language)
                if audio_path:
                    audio_file = open(audio_path, "rb")
                    st.audio(audio_file.read(), format="audio/mp3")
                    audio_file.close()
                    os.remove(audio_path)
    # ASR (record and transcribe)
    # Browser-based audio recording using st_audiorec
    st.info("اضغط زر التسجيل وتحدث لمدة 5 ثوانٍ فقط، ثم توقف عن التسجيل ليتم إرسال الرد تلقائياً.")
    audio_data = st_audiorec()
    if audio_data is not None:
        st.audio(audio_data, format="audio/wav")
        # Save audio_data to a temporary WAV file for transcription
        import tempfile
        import wave
        import numpy as np
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            wf = wave.open(temp_wav.name, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(audio_data)
            wf.close()
            wav_path = temp_wav.name
        with st.spinner(texts["thinking"]):
            transcript = transcribe_audio(wav_path, lang=language)
            # Store the transcript in a pending state to be processed on the next rerun
            st.session_state["pending_transcript"] = transcript
            # Simulate the send button logic
            st.session_state.chat_history.append({"role": "user", "content": transcript})
            prompt_template = conversation_template if language == "ar" else english_conversation_template
            history = "\n".join(
                f"{texts['me'] if m['role']=='user' else texts['teacher']}{m['content']}"
                for m in st.session_state.chat_history if m["role"] != "assistant" or m != st.session_state.chat_history[-1]
            )
            qa_chain = st.session_state.get("qa_chain")
            if qa_chain:
                try:
                    ai_response = qa_chain.run(
                        prompt_template.format(history=history, kid_answer=transcript)
                    )
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    ai_response = f"AI Error: {str(e)}\n{tb}"
            else:
                ai_response = "AI Error: QA chain not initialized."
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            # Always render chat and play TTS for the latest assistant message
            if ai_response:
                st.markdown(f"**{texts['teacher']}** {ai_response}")
                tts_path = text_to_speech(ai_response, lang=language)
                if tts_path:
                    audio_file = open(tts_path, "rb")
                    st.audio(audio_file.read(), format="audio/mp3")
                    audio_file.close()
                    os.remove(tts_path)