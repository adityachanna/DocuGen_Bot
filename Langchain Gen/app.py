import streamlit as st
import os
import pathlib
import tempfile
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_core.messages import AIMessage
from pdf_processor import extract_pdf_content, save_output_document, ensure_directory_exists
from langchain_community.tools.tavily_search import TavilySearchResults
load_dotenv()
UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if UNSTRUCTURED_API_KEY:
    os.environ["UNSTRUCTURED_API_KEY"] = UNSTRUCTURED_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

def process_with_unstructured_api(file_path):
    """Process PDF using Unstructured API with simplified settings"""
    try:
        from langchain_unstructured import UnstructuredLoader
        
        start_time = time.time()
        st.info("Processing with Unstructured API... This may take a minute.")
        loader = UnstructuredLoader(
            file_path=file_path,
            strategy="high-res",  
            partition_via_api=True,
            coordinates=False,  
        )
        
        progress_bar = st.progress(0)
        docs = []
        
        for i, doc in enumerate(loader.lazy_load()):
            docs.append(doc)
            if i % 3 == 0:
                progress = min((i + 1) / 30, 1.0)  
                progress_bar.progress(progress)
        
        progress_bar.progress(1.0)
        extracted_content = "\n\n".join([doc.page_content for doc in docs])
        doc_structure = {}
        for doc in docs:
            page_num = doc.metadata.get("page_number", 0)
            category = doc.metadata.get("category", "Unknown")
            
            if page_num not in doc_structure:
                doc_structure[page_num] = {}
            if category not in doc_structure[page_num]:
                doc_structure[page_num][category] = []
            doc_structure[page_num][category].append(doc.page_content)
        
        processing_time = time.time() - start_time
        st.success(f"Processing completed in {processing_time:.2f} seconds")
        
        return {
            "content": extracted_content,
            "structure": doc_structure
        }
    except Exception as e:
        st.error(f"Error processing with Unstructured API: {e}")
        return None

def main():
    st.title("GenAI Document Generation Bot")
    st.write("Upload a PDF, provide a prompt, and generate a new document with web search capability.")
    
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY environment variable is not set. Please check your .env file.")
        st.stop()
    
    try:
        llm = ChatGroq(
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
            api_key=GROQ_API_KEY,
            temperature=0.4,
        )
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        st.stop()
    temp_dir = os.path.join(os.getcwd(), "temp_files")
    output_dir = os.path.join(os.getcwd(), "output_docs")
    ensure_directory_exists(temp_dir)
    ensure_directory_exists(output_dir)
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.info(f"Temporary file saved: {temp_file_path}")
            try:
                extracted_content = ""
                doc_structure = {}
                
                if UNSTRUCTURED_API_KEY:
                    api_result = process_with_unstructured_api(temp_file_path)
                    if api_result:
                        extracted_content = api_result["content"]
                        doc_structure = api_result["structure"]
                if not extracted_content:
                    extracted_content, _ = extract_pdf_content(temp_file_path)
                
                st.success(f"Successfully extracted content from PDF: {len(extracted_content)} characters")
                if doc_structure:
                    with st.expander("Document Structure"):
                        for page, categories in doc_structure.items():
                            st.write(f"### Page {page}")
                            for category, contents in categories.items():
                                st.write(f"**{category}**: {len(contents)} elements")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                extracted_content = "Error extracting PDF content. Processing with limited information."
            prompt = st.text_area("Enter a prompt:", "Summarize this document")
            use_web_search = st.checkbox("Enable web search for additional context", value=False)
            
            output_format = st.selectbox("Select output format", ["PDF", "DOCX"], index=0)
            
            if st.button("Generate Document"):
                with st.spinner("Processing document... Please wait."):
                    try:
                        web_search_results = ""
                        if use_web_search and TAVILY_API_KEY:
                            with st.spinner("Searching the web for additional context..."):
                                try:
                                    search = TavilySearchResults(max_results=3)
                                    search_results = search.invoke(prompt)
                                    
                                    # Format search results
                                    web_search_results = "\n\nWeb Search Results:\n"
                                    for i, result in enumerate(search_results, 1):
                                        title = result.get('title', 'No title')
                                        content = result.get('content', 'No content')
                                        url = result.get('url', 'No URL')
                                        web_search_results += f"[{i}] {title}: {content}\nSource: {url}\n\n"
                                    
                                    st.info(f"Found {len(search_results)} relevant web results")
                                except Exception as search_error:
                                    st.warning(f"Web search failed: {search_error}")
                                    web_search_results = ""
                        system_message = SystemMessage(
                            content="""You are a document analysis expert. 
                            Analyze the provided PDF content carefully. 
                            Generate a response that maintains appropriate document structure 
                            with headings, paragraphs, and formatting. 
                            Use markdown for formatting with # for headings, 
                            ** for bold text, and * for italic text.
                            If web search results are provided, use them to enhance your analysis."""
                        )
                        max_content_length = 50000
                        truncated_content = extracted_content[:max_content_length]
                        if len(extracted_content) > max_content_length:
                            truncated_content += "\n...[content truncated due to length]..."
                        human_message = HumanMessage(
                            content=f"{prompt}\n\nDocument content:\n{truncated_content}{web_search_results}"
                        )
                        
                        with st.spinner("Generating response with Groq... This may take a minute."):
                            response = llm.invoke([system_message, human_message])
                            generated_text = response.content
                        
                        output_filename = f"{os.path.splitext(os.path.basename(temp_file_path))[0]}_generated.{output_format.lower()}"
                        output_file_path = os.path.join(output_dir, output_filename)
                        
                        save_output_document(generated_text, output_format, output_file_path)
                        
                        st.success(f"Document generated successfully!")
                        
                        with open(output_file_path, "rb") as f:
                            st.download_button(
                                label=f"Download {output_format}",
                                data=f,
                                file_name=output_filename,
                                mime=f"application/{'pdf' if output_format.lower() == 'pdf' else 'vnd.openxmlformats-officedocument.wordprocessingml.document'}"
                            )
                        st.subheader("Generated Content")
                        st.markdown(generated_text)
                        
                    except Exception as e:
                        st.error(f"Error during document generation: {str(e)}")
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
