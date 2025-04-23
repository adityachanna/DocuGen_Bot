# 📄 DocuGen Bot: AI-Powered Document Generation

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)
![Last Updated](https://img.shields.io/badge/Last%20Updated-April%202025-orange.svg)

> **A dual-approach framework for intelligent document analysis and generation, powered by cutting-edge multimodal AI models.**

DocuGen Bot transforms how organizations process and generate documents by combining two complementary AI approaches in one unified platform:

## 🚀 Key Features

- **Dual AI Approach**: Choose between an optimized cost-effective solution or a highly customizable agent-based architecture
- **Multimodal Understanding**: Process text, tables, images, and complex graphics within documents
- **Web-Enhanced Grounding**: Optional integration with search APIs for up-to-date context
- **Format Preservation**: Maintain document structure, headings, and formatting styles
- **Multiple Export Options**: Generate professional PDF or DOCX outputs with proper styling

---

## 🏗️ Architecture Overview

![Architecture Flowchart_ lit py vs Program py(2)](https://github.com/user-attachments/assets/6cefaf4e-0cca-459f-856f-a2595cf35d66)


### Approach 1: Google GenAI (Efficient & Cost-Effective)
> Direct multimodal processing with minimal overhead

### Approach 2: LangChain Agent (Advanced & Customizable)
> Agent-based framework with pluggable tools and rich grounding capabilities

---

## 🔎 Technical Comparison

| Feature | Approach 1: Google GenAI | Approach 2: LangChain Agent |
|---------|--------------------------|------------------------------|
| **Core Engine** | Gemini 2.0 Flash | Llama 4 Maverick (via Groq) |
| **Input Processing** | Direct PDF bytes | Structured PDF partitioning |
| **Content Understanding** | Embedded multimodal | Sequential text + image |
| **External Knowledge** | Google Search API | Tavily Search API |
| **Implementation Complexity** | Low | Medium-High |
| **Customization Potential** | Limited | Extensive |
| **Resource Requirements** | Minimal | Moderate |
| **Total Time** | Faster (5-8s) | Comprehensive (10-14s) |
| **Output Quality** | Good | Excellent |

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.8+
- API keys for chosen approach:
  - **Approach 1**: Google Gemini API key
  - **Approach 2**: Groq API key, Tavily API key (optional), Unstructured API key (optional)

### Quick Start

```bash
# Clone repository
git clone https://github.com/adityachanna/DocuGen_Bot.git
cd DocuGen_Bot

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure your .env file with API keys
echo "GEMINI_API_KEY=your_key_here" > .env
echo "GROQ_API_KEY=your_key_here" >> .env
echo "TAVILY_API_KEY=your_key_here" >> .env
echo "UNSTRUCTURED_API_KEY=your_key_here" >> .env

# Run the application (choose one)
streamlit run "Google Gen/app.py"     # For Google GenAI approach
streamlit run "Langchain Gen/app.py"  # For LangChain approach
```

---

## 🧠 Implementation Details

### Google GenAI Implementation (Approach 1)

```python
def generate_document(client, file_bytes, prompt, model_id="gemini-2.0-flash", use_google_search=True):
    """Generates text content from PDF bytes using Google GenAI with optional search grounding"""
    gen_config = None
    if use_google_search:
        search_tool = {'google_search': {}}
        gen_config = GenerateContentConfig(tools=[search_tool], response_modalities=["TEXT"])
        
    contents_list = [
        types.Part.from_bytes(data=file_bytes, mime_type='application/pdf'),
        prompt
    ]
    
    response = client.models.generate_content(
        model=model_id,
        contents=contents_list,
        config=gen_config
    )
    
    return response.text
```

### LangChain Agent Implementation (Approach 2)

```python
def generate_document(llm, pdf_path, prompt, use_tavily_search=True):
    """Generate content using LangChain agent with PDF analysis and optional web search"""
    pdf_data = process_pdf(pdf_path)
    extracted_text = pdf_data["extracted_text"]
    pdf_base64 = encode_image_to_base64(pdf_data["file_path"])
    
    # Define custom tools
    search = TavilySearchResults(max_results=5)
    
    @tool
    def analyze_pdf(query):
        """Analyzes the PDF document based on the given query"""
        messages = [
            SystemMessage(content="You're an expert at analyzing PDF documents."),
            HumanMessage(content=[
                {"type": "text", "text": f"Analyze this PDF regarding: {query}\n\n{extracted_text[:8000]}..."},
                {"type": "image", "image_url": {"url": f"data:application/pdf;base64,{pdf_base64}"}}
            ])
        ]
        return llm.invoke(messages).content
        
    # Create and execute agent
    agent = create_openai_functions_agent(llm=llm, tools=[search, analyze_pdf], prompt=prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=[search, analyze_pdf], verbose=True)
    
    return agent_executor.invoke({"input": prompt})["output"]
```

---

## 📊 Performance & Evaluation

We evaluated both approaches using a test suite of 50 diverse documents (technical papers, reports, contracts, presentations) with varying complexities:

| Metric | Google GenAI | LangChain Agent |
|--------|-------------|-----------------|
| **Content Accuracy** | Good| Average |
| **Processing Speed** | 0.8s/page | 2s/page |
| **Handling Complex Images** | Good | Moderate |
| **Table Rendering** | Limited | Comprehensive |


---

## 🔧 Usage Guide

### Step 1: Choose Your Approach
Select the appropriate implementation based on your needs:
- **Google GenAI**: For faster, cost-effective processing
- **LangChain Agent**: For maximum customization and accuracy

### Step 2: Document Upload
Upload your PDF document through the Streamlit interface.

### Step 3: Configure Generation Settings
- Enter your generation prompt
- Toggle web search for additional context (recommended for technical documents)
- Select output format (PDF/DOCX)

### Step 4: Generate & Download
Click "Generate Document" and download the resulting file.

---

## 🔍 Project Structure

```
DocuGen_Bot/
├── Google Gen/                    # Approach 1: Google GenAI Implementation
│   ├── app.py                     # Streamlit frontend for Google approach
│   └── genai_utils.py             # Utilities for PDF processing & GenAI
│
├── Langchain Gen/                 # Approach 2: LangChain Agent Implementation
│   ├── app.py                     # Streamlit frontend for LangChain approach
│   └── pdf_processor.py           # Advanced PDF extraction & tool definitions
│
├── output_docs/                   # Generated document outputs
├── temp_files/                    # Temporary file storage
├── .env                           # Environment variables (API keys)
├── requirements.txt               # Project dependencies
└── README.md                      # This documentation
```

---

## 🛠️ Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **OCR Quality** | Implemented tiered extraction with fallbacks from hi-res to basic |
| **Large Document Handling** | Added chunking strategies with overlapping content for context preservation |
| **API Rate Limiting** | Built exponential backoff and request batching |
| **Complex Document Structures** | Enhanced preprocessing to maintain hierarchical document elements |
| **Mixed Content Types** | Developed specialized handlers for tables, charts, and code blocks |

---

## 🔮 Future Development

- [ ] **Fine-tuning Pipeline**: Domain-specific model adaptation for specialized documents
- [ ] **Custom Template Engine**: Pre-built templates for common document types
- [ ] **Multi-Document Analysis**: Compare and synthesize information across multiple PDFs
- [ ] **Interactive Editing**: Web-based interface for post-generation document refinement
---

### Acknowledgments
- Google Gemini API for multimodal document processing
- Groq API for fast LLM inference
- Tavily for web search capabilities
- Unstructured for advanced document parsing

---

<div align="center">
  <p>Developed by <a href="https://github.com/adityachanna">adityachanna</a></p>
  <p>Last updated: April 2025</p>
</div>
