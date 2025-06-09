# 📚 Knowledge Transfer (KT) Project

A robust Knowledge Transfer (KT) system designed to streamline the transition of undocumented and implicit knowledge across teams using AI-powered techniques. This solution ensures continuity, accuracy, and ease of onboarding through automated parsing, semantic search, Q&A, and summarization.

---

## 🚀 Project Highlights

- ✅ **Multi-format Ingestion**: Supports Word, PDF, Excel, PowerPoint, and text files.
- 🔍 **Semantic Search**: Find answers even when phrased differently using vector-based retrieval.
- 🤖 **LLM-Powered Q&A**: Uses GPT-based models to answer questions from the indexed knowledge base.
- 📊 **Tiered Chunking**: Implements hierarchical chunking (Tier 1 & Tier 2) for better context preservation.
- 🧠 **Few-shot Prompting**: Guides the model with custom examples to reduce hallucinations.
- 💾 **Document-level Metadata**: Tracks source, author, and context for every chunk.
- 🛡️ **RAG (Retrieval-Augmented Generation)**: Ensures responses are grounded in actual indexed content.
- 🧪 **Validation Metrics**: Measures hallucination rate, token usage, and retrieval accuracy.

---

## 🗂️ Folder Structure
KT-Project/
│
├── ingestion/ # Parsing and pre-processing scripts
│ ├── extract_pdf_text.py
│ ├── extract_docx_text.py
│ └── extract_pptx_text.py
│
├── indexing/ # Tiered chunking and vector embedding
│ ├── tier1_chunker.py
│ ├── tier2_chunker.py
│ └── vector_store.py
│
├── qa_engine/ # LLM-powered Q&A system
│ ├── rag_pipeline.py
│ ├── prompt_templates.py
│ └── validate_response.py
│
├── streamlit_app/ # Frontend interface
│ └── app.py
│
├── data/ # Sample documents
│
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## ⚙️ Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-repo/kt-project.git
   cd kt-project
Create and activate virtual environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit App

bash
Copy
Edit
cd streamlit_app
streamlit run app.py
🔧 Tech Stack
Python – Core scripting

Streamlit – Frontend interface

LangChain / LlamaIndex – RAG & retrieval logic

OpenAI GPT-4 / Gemini / Claude – LLM integrations

FAISS / Weaviate / Astra DB – Vector search

Azure Blob / GCS / Local FS – File storage

📈 Evaluation Metrics
Metric	Description
Hallucination Rate	% of answers not found in original docs
Retrieval Accuracy	% of correct chunk matches
Token Usage	Average input/output tokens per query
Latency	Time taken per complete response cycle

👨‍💻 Use Case Scenarios
🔄 Project Handover

🧑‍🏫 Employee Onboarding

📚 Document Understanding

🧠 Long-Term Knowledge Preservation

📝 Future Improvements
 Fine-tune LLM on internal domain knowledge

 Add feedback loop to train on user corrections

 Integrate Slack/Teams for inline Q&A

 Multi-language support (Hindi, Tamil, etc.)

🙌 Acknowledgements
Special thanks to all contributors, knowledge holders, and AI platforms (OpenAI, Hugging Face, etc.) that made this project possible.



