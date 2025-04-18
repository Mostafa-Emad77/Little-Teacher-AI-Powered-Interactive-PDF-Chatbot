# Little Teacher: AI-Powered Interactive PDF Chatbot

Little Teacher is an interactive Streamlit web app that transforms any PDF into a child-friendly, AI-powered learning experience. Upload a PDF, and the app will guide children through the content with engaging questions, explanations, and voice interaction in both Arabic and English.

## Features

- **PDF Upload & Viewing:** Upload any PDF and view it page by page.
- **AI Chatbot:** Ask and answer questions about the PDF content using advanced retrieval-augmented generation (RAG) with vector search.
- **Multilingual Support:** Switch between Arabic and English for both the interface and AI responses.
- **Text-to-Speech (TTS):** All AI responses are read aloud for accessibility and engagement.
- **Speech-to-Text (ASR):** Answer questions by speaking—your voice is transcribed and sent to the AI.
- **Child-Friendly Prompts:** The AI uses special prompts to explain and quiz in a way that's fun and easy for kids.
- **Streamlit Community Cloud Ready:** Deploy with one click—no server setup required.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Mostafa-Emad77/little-teacher.git
cd little-teacher
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Set API Keys

This app requires API keys for:
- **HuggingFace Inference API** (for speech-to-text)
- **OpenRouter/OpenAI** (for LLM responses)

On Streamlit Community Cloud, add these as secrets in the app settings:

```
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
OPENROUTER_API_KEY=your_openrouter_key
```

Or, set them as environment variables locally.

### 4. Run the App Locally

```bash
streamlit run app.py
```

### 5. Deploy on Streamlit Community Cloud

1. Push your code to GitHub.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click "New app", select your repo and app.py, and deploy!

## Usage

1. **Upload a PDF** using the sidebar.
2. **Read and interact**: The AI will introduce the topic and ask questions.
3. **Reply by text or voice**: Type your answer or use the microphone.
4. **Listen to responses**: The AI will read its replies aloud.
5. **Switch languages**: Use the sidebar to toggle between Arabic and English.

## Technologies Used

- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [HuggingFace Inference API](https://huggingface.co/inference-api)
- [OpenRouter](https://openrouter.ai/) (or OpenAI-compatible LLMs)
- [st_audiorec](https://github.com/stefanrmmr/st_audiorec) (browser-based audio recording)
- [gTTS](https://pypi.org/project/gTTS/) (text-to-speech)
- [PyMuPDF](https://pymupdf.readthedocs.io/) (PDF processing)
- [FAISS](https://github.com/facebookresearch/faiss) (vector search)

## Credits

- Prompt templates and educational logic inspired by [AraTec.py](./AraTec.py).
- Built by [Your Name] for educational and research purposes.

## License

This project is licensed under the MIT License.

---

Enjoy learning with Little Teacher!
