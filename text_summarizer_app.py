import streamlit as st


st.set_page_config(page_title="Text Summarizer", layout="centered")

from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from deep_translator import GoogleTranslator
import docx
import fitz
from textblob import TextBlob
import textstat
from langdetect import detect
import pandas as pd
import nltk
import ssl
import torch
import pyperclip
import base64


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


@st.cache_resource
def load_summarizer():
    try:
        model_name = "sshleifer/distilbart-cnn-12-6"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return pipeline("summarization", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        st.info("Please make sure you have a stable internet connection and try again.")
        return None


summarizer = load_summarizer()

def get_text_metrics(text):
    """Calculate various text metrics"""
    metrics = {
        'word_count': len(text.split()),
        'sentence_count': len(TextBlob(text).sentences),
        'flesch_score': textstat.flesch_reading_ease(text),
        'smog_score': textstat.smog_index(text),
        'language': detect(text)
    }
    return metrics

def translate_text(text, target_lang):
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

def read_file(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "application/pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return ""


st.title("📝TinyTalk- Your Text summarizer and Translator")

input_mode = st.radio("Choose input type:", ["Manual Text", "Upload Document"])

text = ""
if input_mode == "Manual Text":
    text = st.text_area("Enter text to summarize:")
else:
    file = st.file_uploader("Upload a file (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
    if file:
        text = read_file(file)
        st.success("Document loaded successfully!")


min_len = st.slider("Minimum summary length", 10, 100, 30)
max_len = st.slider("Maximum summary length", 30, 300, 60)
target_lang = st.selectbox("Translate summary to:", ["en", "ta", "hi", "fr", "es"])


if st.button("Summarize"):
    if not text.strip():
        st.warning("Please enter or upload some text first.")
    elif summarizer is None:
        st.error("Model failed to load. Please check your internet connection and try again.")
    else:
        with st.spinner("Analyzing and summarizing..."):
            try:
                
                original_metrics = get_text_metrics(text)
                
               
                translated_input = translate_text(text, "en")
                summary = summarizer(translated_input, max_length=max_len, min_length=min_len, do_sample=False)
                result = summary[0]['summary_text']
                translated_summary = translate_text(result, target_lang)
                
                
                summary_metrics = get_text_metrics(result)
                
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Original Text Metrics")
                    st.write(f"Word Count: {original_metrics['word_count']}")
                    st.write(f"Sentence Count: {original_metrics['sentence_count']}")
                    st.write(f"Flesch Readability Score: {original_metrics['flesch_score']:.1f}")
                    st.write(f"SMOG Index: {original_metrics['smog_score']:.1f}")
                    st.write(f"Detected Language: {original_metrics['language']}")
                
                with col2:
                    st.subheader("📊 Summary Metrics")
                    st.write(f"Word Count: {summary_metrics['word_count']}")
                    st.write(f"Sentence Count: {summary_metrics['sentence_count']}")
                    st.write(f"Flesch Readability Score: {summary_metrics['flesch_score']:.1f}")
                    st.write(f"SMOG Index: {summary_metrics['smog_score']:.1f}")
                    st.write(f"Detected Language: {summary_metrics['language']}")
                
               
                compression_ratio = (1 - (summary_metrics['word_count'] / original_metrics['word_count'])) * 100
                st.metric("Compression Ratio", f"{compression_ratio:.1f}%")
                
                st.subheader("📝 Summary:")
                st.write(translated_summary)
                
                
                col1, col2 = st.columns(2)
                
                with col1:
                    
                    st.text_area("Copy from here:", translated_summary, height=100, key="copy_area")
                    
                
                with col2:
                    
                    def get_download_link(text, filename):
                        b64 = base64.b64encode(text.encode()).decode()
                        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">💾 Download Summary</a>'
                        return href
                    
                    st.markdown(get_download_link(translated_summary, "summary.txt"), unsafe_allow_html=True)
                    st.info("Click the link above to download the summary")
            except Exception as e:
                st.error(f"An error occurred during summarization: {str(e)}")
                st.info("Please try again with a different text or check your internet connection.")
