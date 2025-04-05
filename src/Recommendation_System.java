import streamlit as st
import requests
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer

# Load models once
tfidf_vectorizer = TfidfVectorizer()
count_vectorizer = CountVectorizer()
hashing_vectorizer = HashingVectorizer(n_features=5000)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
word2vec_model = None  # Lazy initialization to avoid overhead

def get_embeddings(texts):
    global word2vec_model
    tokenized_texts = [text.split() for text in texts]
    if word2vec_model is None:
        word2vec_model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
    embeddings = [np.mean([word2vec_model.wv[word] for word in text if word in word2vec_model.wv], axis=0) if text else np.zeros(100) for text in tokenized_texts]
    return np.array(embeddings)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_arxiv_papers(query, max_results=10):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

def parse_arxiv_response(xml_response):
    root = ET.fromstring(xml_response)
    ns = {'ns': 'http://www.w3.org/2005/Atom'}
    papers = []
    for entry in root.findall('ns:entry', ns):
        papers.append({
            'title': preprocess_text(entry.find('ns:title', ns).text),
            'summary': entry.find('ns:summary', ns).text.strip(),  # Full summary
            'link': entry.find('ns:id', ns).text,
            'author': entry.find('ns:author/ns:name', ns).text if entry.find('ns:author/ns:name', ns) is not None else 'Unknown',
            'published': entry.find('ns:published', ns).text if entry.find('ns:published', ns) is not None else 'Unknown'
        })
    return papers

def recommend_papers(query, papers, model_type):
    corpus = [preprocess_text(query)] + [paper['summary'] for paper in papers]
    
    if model_type == "TF-IDF":
        matrix = tfidf_vectorizer.fit_transform(corpus)
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    elif model_type == "BOW":
        matrix = count_vectorizer.fit_transform(corpus)
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    elif model_type == "Hashing Vectorizer":
        matrix = hashing_vectorizer.fit_transform(corpus)
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    elif model_type == "Word2Vec":
        embeddings = get_embeddings(corpus)
        similarities = cosine_similarity([embeddings[0]], embeddings[1:]).flatten()
    elif model_type == "SBERT":
        embeddings = sbert_model.encode(corpus)
        similarities = cosine_similarity([embeddings[0]], embeddings[1:]).flatten()
    else:
        return []  # Handle unknown model types
    
    return sorted(zip(papers, similarities), key=lambda x: x[1], reverse=True)

def main():
    st.title("📄 Research Paper Recommendation System")
    query = st.text_input("🔍 Enter Research Topic:")
    model_choice = st.selectbox("📊 Choose Model:", ["TF-IDF", "BOW", "Word2Vec", "Hashing Vectorizer", "SBERT"])
    
    if st.button("Find Papers"):
        xml_response = fetch_arxiv_papers(query)
        if xml_response:
            papers = parse_arxiv_response(xml_response)
            recommendations = recommend_papers(query, papers, model_choice)
            
            st.subheader("📌 Recommended Papers:")
            for paper, score in recommendations:
                st.markdown(f"### [{paper['title']}]({paper['link']})")
                st.markdown(f"*Author:* {paper['author']}")
                st.markdown(f"*Published Date:* {paper['published']}")
                st.markdown(f"*Summary:* {paper['summary']}")
                st.markdown(f"*Relevance Score:* {score:.4f}")
                st.write("---")
        else:
            st.error("⚠ Failed to fetch papers. Try again later.")

if _name_ == "_main_":
    main()
