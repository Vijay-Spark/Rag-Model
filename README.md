# 📚 Simple RAG Document Question Answering System

A beginner-friendly Retrieval-Augmented Generation (RAG) system running in **Google Colab** and on **Localhost**.

## 🌐 Running on Localhost (Web UI)
The web application is running live on:
👉 **[http://localhost:8501](http://localhost:8501)**

To start or restart the web server anytime (using Python 3.13):
```powershell
python -m streamlit run app.py --server.port 8501
```

### Features in the Web App
1. **Interactive Chat**: Ask questions about your documents in real-time.
2. **File Uploader**: Drag and drop single or multiple PDF documents.
3. **One-Click Sample Loader**: Test immediately using the built-in `sample_ai_overview.pdf`.
4. **Source Attribution**: Inspect retrieved chunks with page numbers, filenames, and cosine similarity scores.
5. **Configurable Sliders**: Adjust Chunk Size, Chunk Overlap, and Top-k retrieval dynamically.
6. **Gemini API Key Field**: Enter or update your API key directly in the sidebar.

---

## 📁 Project Files
- **`app.py`**: Streamlit interactive web application (`http://localhost:8501`).
- **`Simple_RAG_Document_QA.ipynb`**: The 9-cell Jupyter Notebook for Google Colab.
- **`rag_qa_system.py`**: Terminal / CLI Python version.
- **`sample_ai_overview.pdf`**: Built-in test PDF document.
- **`.env`**: Stores the local `GEMINI_API_KEY`.
"# Rag-Model" 
