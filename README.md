#  Multi-Agent AI Business Intelligence System (RAG-Powered)

An advanced **AI-powered Business Intelligence & Decision Support System** built using **Multi-Agent Architecture**, **RAG (Retrieval-Augmented Generation)**, and modern LLM frameworks.

This project transforms raw data into **actionable insights, dashboards, and strategic recommendations** using collaborative AI agents.

---

##  Overview

Instead of relying on a single AI model, this system simulates a real-world analytics team composed of specialized AI agents that work together to:

- Retrieve relevant knowledge from documents
- Analyze structured and unstructured data
- Summarize complex reports
- Generate business recommendations
- Coordinate the entire workflow

---

##  System Architecture

The system is built around a **Multi-Agent Framework**:

-  **Retriever Agent** → Fetches relevant information from documents & vector database  
-  **Analyst Agent** → Detects trends, patterns, and insights from data  
-  **Summarizer Agent** → Converts long reports into concise summaries  
-  **Recommender Agent** → Produces strategic, data-driven business decisions  
-  **Supervisor Agent** → Orchestrates and manages all agent interactions  

---

##  Key Features

-  RAG pipeline with vector database integration  
-  Short-term + long-term semantic memory system  
-  Automatic dashboard generation from CSV/Excel uploads  
-  Interactive visualizations using Plotly  
-  Chat-based AI interface (Streamlit UI)  
-  Full support for Arabic & English  
-  Upload support for PDF, CSV, and Excel files  
-  High-speed inference using Groq + LLaMA 3.3 70B  

---

##  Tech Stack

- LangGraph (Multi-Agent Orchestration)
- LangChain
- Chroma Vector Database
- Streamlit
- Groq API (LLaMA 3.3 70B)
- Plotly
- Python

---

##  Project Structure (Example)
project/
│
├── app/
│ ├── agents/
│ ├── memory/
│ ├── tools/
│ ├── ui/
│ └── main.py
│
├── data/
├── vectorstore/
├── requirements.txt



---

##  Installation & Setup

### Prerequisites
- Python 3.11+
- pip
- A free [Groq API key](https://console.groq.com/keys) 

### 1. Clone the repository
```bash
git clone https://github.com/mahmoudtamer704/multi-agent-project.git
cd multi-agent-project/project
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file inside the `project/` folder:
```env
# Choose at least one LLM provider
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile


```

### 5. Run the app locally
```bash
streamlit run app.py
```
The app will open automatically at `http://localhost:8501`.

### 6. Using the app
- Go to the ** Upload Data** panel in the sidebar and upload a CSV, Excel, PDF, Word, or TXT file.
- Use the ** Dashboard** tab for auto-generated KPIs and charts.
- Use the ** Chat** tab to ask questions in English or Arabic — the Orchestrator routes each query to the right agent (RAG, Analytics, Web Search, or Combined).
- Use the ** Stats** tab for a detailed statistical summary of the uploaded dataset.

---

##  Deployment

This project is deployed using **Streamlit** (Streamlit Community Cloud). To deploy your own copy:
1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Select this repository, set the main file path to `project/app.py`.
4. Add your `GROQ_API_KEY` (and any other provider keys) under **Secrets** in the Streamlit Cloud app settings.
5. Deploy — Streamlit Cloud will install `requirements.txt` and launch the app automatically.

---


