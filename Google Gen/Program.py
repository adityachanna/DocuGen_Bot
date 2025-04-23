import streamlit as st
import os
from dotenv import load_dotenv
import pathlib
from genai_utils import (
    load_genai_client,
    process_pdf,
    generate_document,
    save_output_document
)
load_dotenv()

def main():
    st.title("Gen AI Document Generator")
    st.write("Upload a PDF, provide a prompt, and generate a new document.")
    try:
        client = load_genai_client()
    except ValueError as e:
        st.error(f"Failed to initialize GenAI Client: {e}")
        st.stop() 
    except Exception as e:
        st.error(f"An unexpected error occurred during client initialization: {e}")
        st.stop()

    uploaded_file = st.file_uploader("1. Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        temp_dir = pathlib.Path("temp_files")
        temp_dir.mkdir(exist_ok=True) 
        temp_file_path = temp_dir / uploaded_file.name
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.info(f"Temporary file saved: {temp_file_path}")
            try:
                file_bytes = process_pdf(temp_file_path)
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                if temp_file_path.exists():
                    os.remove(temp_file_path)
                st.stop()

            prompt = st.text_input("2. Enter a prompt:", "Summarize this document")
            output_format = st.selectbox("3. Select output format", ["PDF", "DOCX"], index=1)
            use_grounding = st.checkbox("Use Google Search for grounding?", value=True)

            if st.button("4. Generate Document"):
                with st.spinner("Generating document... Please wait."):
                    try:
                    
                        generated_text = generate_document(
                            client=client,
                            file_bytes=file_bytes,
                            # Update the prompt in Program.py
                            prompt = "Analyze the provided PDF document. " + \
                            "First, search the web for the most current information about the topics in this document. " + \
                            "Then, generate a response that closely matches the original document's formatting, " + \
                            "structure (headings, lists, paragraphs), and writing style. " + prompt,
                            use_google_search=use_grounding
                        )
                        
                        output_dir = pathlib.Path("output_docs")
                        output_filename = f"{temp_file_path.stem}_generated.{output_format.lower()}"
                        output_file_path = output_dir / output_filename

                        save_output_document(generated_text, output_format, output_file_path)

                        st.success(f"Document generated successfully!")
                        with open(output_file_path, "rb") as f:
                            st.download_button(
                                label=f"Download {output_format}",
                                data=f,
                                file_name=output_filename,
                                mime=f"application/{'pdf' if output_format.lower() == 'pdf' else 'vnd.openxmlformats-officedocument.wordprocessingml.document'}"
                            )

                    except ImportError as e:
                         st.error(f"Missing Library: {e}. Please install required libraries (e.g., pip install python-docx fpdf2).")
                    except Exception as e:
                        st.error(f"Error during document generation or saving: {e}")

        finally:
             if temp_file_path.exists():
                 try:
                     os.remove(temp_file_path)
                     st.info(f"Cleaned up temporary file: {temp_file_path}")
                 except Exception as e:
                     st.warning(f"Could not remove temporary file {temp_file_path}: {e}")
if __name__ == "__main__":
    main()