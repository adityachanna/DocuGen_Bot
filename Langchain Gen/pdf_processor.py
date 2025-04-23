import os
import pathlib
import tempfile
from typing import Optional, Tuple, List
import base64
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition
from unstructured.staging.base import elements_to_text

from langchain_groq import ChatGroq  
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")
def ensure_directory_exists(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

ensure_directory_exists(os.path.join(os.getcwd(), "temp_files"))
ensure_directory_exists(os.path.join(os.getcwd(), "output_docs"))
def extract_text_without_ocr(file_path):
    """Extract text from PDF without using OCR (doesn't require Tesseract)"""
    try:
        elements = partition_pdf(
            file_path,
            extract_images_in_pdf=True,  
            infer_table_structure=True,  
            strategy="hi_res",  
            include_page_breaks=True,
            ocr_languages=None,
        )
        return elements_to_text(elements)
    except Exception as e:
        print(f"Error in simple PDF extraction: {e}")
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    text += "[Page contains no extractable text]\n"
            return text if text.strip() else "No text could be extracted from this PDF."
        except Exception as e2:
            print(f"Error in PyPDF2 extraction: {e2}")
            return "Error extracting text from PDF. Consider installing PyPDF2 with 'pip install PyPDF2'."

def load_langchain_client():
    """Initialize and return the Groq model"""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    
    return ChatGroq(
        api_key=api_key,
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.4,
    )

def encode_image_to_base64(image_path):
    """Convert an image file to a base64 encoded string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_pdf(file_path):
    """Process PDF using Unstructured for text extraction with OCR capabilities"""
    try:
     
        if isinstance(file_path, str):
            if not os.path.isabs(file_path):
                abs_path = os.path.abspath(file_path)
                if not os.path.exists(abs_path):
                    temp_path = os.path.join(os.getcwd(), "temp_files", os.path.basename(file_path))
                    if os.path.exists(temp_path):
                        file_path = temp_path
                    else:
                        raise FileNotFoundError(f"File '{file_path}' not found at '{abs_path}' or '{temp_path}'!")
                else:
                    file_path = abs_path
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File '{file_path}' not found!")
        if isinstance(file_path, bytes):
            temp_dir = os.path.join(os.getcwd(), "temp_files")
            ensure_directory_exists(temp_dir)
            temp_file_path = os.path.join(temp_dir, f"temp_upload_{os.urandom(4).hex()}.pdf")
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_path)
            file_path = temp_file_path
            print(f"Created temporary file: {file_path}")
        
        try:
            elements = partition_pdf(
                file_path, 
                extract_images_in_pdf=True,
                infer_table_structure=True,
                strategy="hi_res",  
                vision_model="chipper",  
                
                include_page_breaks=True,
            )
            extracted_text = elements_to_text(elements)
        except Exception as ocr_error:
            if "tesseract is not installed" in str(ocr_error):
                print("Tesseract OCR not available. Falling back to basic PDF extraction...")
                extracted_text = extract_text_without_ocr(file_path)
                elements = None
            else:
                raise
        
        return {
            "file_path": file_path,
            "extracted_text": extracted_text,
            "elements": elements if 'elements' in locals() else None
        }
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise

def extract_pdf_content(file_path) -> Tuple[str, List[str]]:
    """
    Extract text and images from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Tuple of (extracted_text, list_of_base64_encoded_images)
    """
    try:
        pdf_data = process_pdf(file_path)
        extracted_text = pdf_data["extracted_text"]
      
        images = []
        try:

            pdf_base64 = encode_image_to_base64(pdf_data["file_path"])
            images.append(pdf_base64)
        except Exception as e:
            print(f"Error encoding PDF as image: {e}")
        
        return extracted_text, images
    except Exception as e:
        print(f"Error in extract_pdf_content: {e}")
        raise

def generate_document(
    llm, 
    pdf_path, 
    prompt: str, 
    use_tavily_search: bool = True,
    tavily_api_key: Optional[str] = None
):
    """Generate content based on PDF using extracted text and optional Tavily search grounding"""
    
    if not pdf_path:
        raise ValueError("No PDF provided for processing")
    
    system_prompt = """You are an AI document generator expert at analyzing documents.
    Analyze the provided document content in its entirety and generate a response that maintains 
    the original document's formatting, structure (headings, lists, paragraphs), and writing style.
    Include all relevant information from the document in your response.
    """
    pdf_data = process_pdf(pdf_path) if isinstance(pdf_path, str) else pdf_path
    pdf_base64 = encode_image_to_base64(pdf_data["file_path"])
    extracted_text = pdf_data["extracted_text"]
    
    try:
        if use_tavily_search:
            if not tavily_api_key and not os.getenv("TAVILY_API_KEY"):
                raise ValueError("Tavily API key is required for search functionality")
            
            if tavily_api_key:
                os.environ["TAVILY_API_KEY"] = tavily_api_key
                
            search = TavilySearchResults(max_results=5)
            @tool
            def analyze_pdf(query: str) -> str:
                """Analyzes the PDF document based on the given query"""
                messages = [
                    SystemMessage(content="You're an expert at analyzing PDF documents."),
                    HumanMessage(content=[
                        {"type": "text", "text": f"Analyze this PDF document regarding: {query}\n\nExtracted text: {extracted_text[:8000]}..."},
                        {"type": "image", "image_url": {"url": f"data:application/pdf;base64,{pdf_base64}"}}
                    ])
                ]

                response = llm.invoke(messages)
                return response.content
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
                ("ai", "{agent_scratchpad}")
            ])
            
            agent = create_openai_functions_agent(
                llm=llm,
                tools=[search, analyze_pdf],
                prompt=prompt_template
            )
           
            agent_executor = AgentExecutor(
                agent=agent,
                tools=[search, analyze_pdf],
                verbose=True
            )
            
            result = agent_executor.invoke({"input": prompt})
            return result["output"]
        else:
            return fallback_direct_approach(llm, system_prompt, prompt, pdf_base64, extracted_text)
    except Exception as e:
        print(f"Agent execution failed: {e}. Falling back to direct approach.")
        return fallback_direct_approach(llm, system_prompt, prompt, pdf_base64, extracted_text)

def fallback_direct_approach(llm, system_prompt, prompt, pdf_base64, extracted_text):
    """Direct approach without using agent - fallback method"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=[
            {"type": "text", "text": f"{prompt}\n\nExtracted document text:\n{extracted_text[:8000]}..."},
            {"type": "image", "image_url": {"url": f"data:application/pdf;base64,{pdf_base64}"}}
        ])
    ]
    
    response = llm.invoke(messages)
    return response.content

def save_output_document(content, output_format, output_path):
    """Reuse the existing save function from genai_utils.py"""
    from genai_utils import save_output_document as original_save
    return original_save(content, output_format, output_path)
