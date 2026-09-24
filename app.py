import os
import shutil
import platform
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from google import genai
from src.modules.extractor import extract_pdf_multimodal

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="Glass Data Extraction Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Glass Science Literature Extraction Pipeline")
st.markdown("Extract standardized glass compositions, kinetics, and stress profile data into your master dataset.")

# Initialize Gemini Client safely
if not api_key:
    st.error("❌ GEMINI_API_KEY not found in environment variables. Check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

# Sidebar Configuration for Directory Scanning & Page Slicing
st.sidebar.header("⚙️ Extraction Settings")

# Toggle between Papers and Patents
doc_source = st.sidebar.radio("Select Document Type:", ["Literature Papers", "Patents"])

if doc_source == "Literature Papers":
    raw_dir = "data/01_raw_papers"
    processed_dir = "data/02_processed_papers"
else:
    raw_dir = "data/02_raw_patents"
    processed_dir = "data/03_processed_patents"

# Ensure directories exist
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

# Automatically scan the selected directory for PDF files
available_pdfs = [f for f in os.listdir(raw_dir) if f.lower().endswith(".pdf")]

if not available_pdfs:
    st.sidebar.warning(f"⚠️ No PDF files found in `{raw_dir}`. Please place your PDFs there.")
    selected_pdf_name = None
else:
    selected_pdf_name = st.sidebar.selectbox("Select File from Local Directory", available_pdfs)

start_page = st.sidebar.number_input("Start Page (Token Saver)", min_value=1, value=1)
end_page = st.sidebar.number_input("End Page (Token Saver)", min_value=1, value=20)

if selected_pdf_name:
    local_pdf_path = os.path.join(raw_dir, selected_pdf_name)

    st.sidebar.success(f"📄 Target File:\n`{local_pdf_path}`")

    # Button to open the local PDF automatically on the host machine
    if st.sidebar.button("📂 Open PDF in Desktop Viewer"):
        try:
            if platform.system() == "Windows":
                os.startfile(local_pdf_path)
            elif platform.system() == "Darwin":  # macOS
                os.system(f"open '{local_pdf_path}'")
            else:  # Linux
                os.system(f"xdg-open '{local_pdf_path}'")
            st.sidebar.success("✅ Launched local PDF viewer!")
        except Exception as e:
            st.sidebar.error(f"Could not open file automatically: {e}")

    if st.sidebar.button("🚀 Extract Data from PDF", type="primary"):
        with st.spinner("Analyzing document via multimodal vision and slicing token payload..."):
            extracted_data, error_msg = extract_pdf_multimodal(
                client=client,
                pdf_path=local_pdf_path,
                start_page=start_page,
                end_page=end_page
            )

        if error_msg:
            st.error(error_msg)
        elif not extracted_data:
            st.warning("⚠️ No structured experiment records were returned by the model.")
        else:
            st.success(f"✨ Successfully extracted {len(extracted_data)} experimental records!")

            df_temp = pd.DataFrame(extracted_data)
            # Convert boolean columns to integer type to prevent NaN dtype conflicts during editing/row deletion
            for col in df_temp.select_dtypes(include=['bool']).columns:
                df_temp[col] = df_temp[col].astype(int)

            # Store in session state
            st.session_state['extracted_df'] = df_temp
            st.session_state['current_pdf'] = selected_pdf_name
            st.session_state['current_pdf_path'] = local_pdf_path

# Main Panel Display & Editable Approval Workflow
if 'extracted_df' in st.session_state and st.session_state['extracted_df'] is not None:
    st.subheader(f"📊 Editable Verification Panel for: `{st.session_state['current_pdf']}`")
    st.info(
        "💡 **Note:** You can click directly into any cell, add rows, or delete/cut rows using the data editor below before approving.")

    # st.data_editor makes the dataframe interactive and returns the modified DataFrame back to us
    edited_df = st.data_editor(
        st.session_state['extracted_df'],
        use_container_width=True,
        num_rows="dynamic",
        key="glass_data_editor"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Approve & Append to Master Dataset", type="primary"):
            master_path = r"D:\post doc\dataset\Dataset_V8.xlsx"
            try:
                # Use the edited_df so all user modifications are preserved
                if os.path.exists(master_path):
                    df_master = pd.read_excel(master_path)
                    df_updated = pd.concat([df_master, edited_df], ignore_index=True)
                else:
                    df_updated = edited_df

                os.makedirs(os.path.dirname(master_path), exist_ok=True)
                df_updated.to_excel(master_path, index=False)

                # Automatically move PDF from raw to processed folder
                dest_path = os.path.join(processed_dir, st.session_state['current_pdf'])
                if os.path.exists(st.session_state['current_pdf_path']):
                    shutil.move(st.session_state['current_pdf_path'], dest_path)

                st.success(
                    f"✅ Approved! Appended modified records to master dataset and moved file to `{processed_dir}`.")

                # Clear session state so UI resets for the next paper
                del st.session_state['extracted_df']
                del st.session_state['current_pdf']
                st.rerun()

            except Exception as e:
                st.error(f"Failed during approval workflow: {e}")

    with col2:
        if st.button("🗑️ Discard / Re-extract"):
            if 'extracted_df' in st.session_state:
                del st.session_state['extracted_df']
            st.rerun()

else:
    if not available_pdfs:
        st.info(f"👈 Please add your PDF documents into `{raw_dir}` to begin extraction.")
    else:
        st.info("👈 Select a document from the sidebar and click **'Extract Data from PDF'** to begin verification.")