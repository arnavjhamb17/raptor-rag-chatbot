import streamlit as st
from bfs_backtrack import bfs_search
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import io
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import fitz  # PyMuPDF
import docx
import tempfile
import json
import re
from datetime import datetime
import numpy as np
import cv2

# Load and verify environment variables
load_dotenv()
required_vars = ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_DEPLOYMENT_NAME']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Simplified OCR Extraction Prompt - NO TDS Classification
INVOICE_EXTRACTION_PROMPT = """
You are an expert in analyzing invoices and business documents. Extract the following information from the provided text:

Invoice/Document Text:
{invoice_text}

Please extract and respond in EXACTLY this JSON format:

{{
    "state": "State where transaction occurred",
    "city": "City where transaction occurred", 
    "amount": "Transaction amount (numerical value with currency)",
    "payment_type": "Clear, detailed description of the goods/services being paid for",
    "vendor_name": "Name of the vendor/service provxider",
    "invoice_number": "Invoice or bill number",
    "date": "Invoice or transaction date",
    "confidence": "Confidence percentage (0-100)"
}}

Focus on:
- Look for location details (billing address, service location, etc.)
- Identify the exact nature of goods/services (be specific - e.g., "Software development services", "Equipment rental", "Professional consultation")
- Extract transaction amount with currency
- Find vendor/company details
- Extract invoice number and date if available

If information is not clearly available, use "Not found" for that field.
Do NOT attempt to classify TDS sections - only extract factual information from the document.
"""

# Enhanced TDS Analysis Prompt that uses extracted invoice data
TDS_ANALYSIS_PROMPT = """
Role: You are a tax professional who is an expert in TDS provisions under the Indian Income Tax Act.

IMPORTANT: Respond ONLY in the JSON format shown below for TDS section/rate queries.

User Query: {query}
Context Chunks from TDS Knowledge Base:
{context}

Invoice/Document Information (if available):
{invoice_info}

Based on the context and invoice details, analyze the TDS requirements and provide the response in EXACTLY this JSON format:

{{
    "analysis_results": [
        {{
            "goods_or_services": "Specific description from invoice or query",
            "explanation": "Brief explanation of the transaction nature",
            "primary_section": "194X",
            "primary_rate": "X% for Individual/HUF, Y% for Others",
            "primary_explanation": "Detailed justification for primary section based on legal provisions",
            "alternate_section": "194Y or 'None'",
            "alternate_rate": "X% or 'N/A'",
            "alternate_explanation": "Detailed justification for alternate section or 'N/A'",
        }}
    ],
    "overall_confidence": 85,
    "sources": ["Referenced document from knowledge base"],
    "additional_notes": "Any additional considerations, exemptions, or special cases"
}}

Use the invoice information to understand the exact nature of goods/services and determine the most appropriate TDS section based on the legal provisions in the context.
"""

GENERAL_TDS_PROMPT = """
Role: You are a knowledgeable tax professional expert in TDS provisions under the Indian Income Tax Act.

IMPORTANT RESTRICTIONS:
1. If the query is unrelated to TDS/taxes, respond: "I don't have enough knowledge to answer that question. However, I'm here to help with any TDS-related queries you might have!"
2. For simple conversational queries, respond naturally and offer help with TDS.
3. Only answer TDS-related questions using provided context.
4. Provide responses in clear paragraph format, NOT in JSON or table format.

Context from TDS Knowledge Base:
{context}

Invoice/Document Information (if available):
{invoice_info}

Conversation History:
{conversation_history}

User Query: {query}

Please provide a helpful and accurate response in paragraph format based on the context and your expertise.
If invoice information is available, use it to provide more specific guidance.
"""

NON_TDS_DETECTION_PROMPT = """
Analyze this query and determine if it's related to TDS, taxation, compliance, or accounting.

Query: {query}

Respond with one word:
- "TDS" if related to TDS/taxation
- "CONVERSATIONAL" if a simple greeting
- "UNRELATED" if completely unrelated
Response:
"""

# Azure OpenAI config
llm = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_OPENAI_VERSION", "2024-10-21"),
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0.2,
)

def get_context_for_query(query, max_docs=20, threshold=0.5):
    docs = bfs_search(query, threshold=threshold, max_docs=max_docs)
    context = []
    for doc in docs:
        meta = doc.metadata
        ref = f"{meta.get('source_file','unknown')} (chunk {meta.get('chunk_index',meta.get('cluster','-'))})"
        context.append(f"[{ref}]: {doc.page_content[:500]}")
    return "\n\n".join(context)

def is_tds_rate_section_query(query):
    rate_keywords = ['rate','section','applicable','deduction','percentage','%','tds for','which section']
    service_keywords = ['service','goods','payment','transaction','invoice','bill']
    q = query.lower()
    return any(k in q for k in rate_keywords) or any(k in q for k in service_keywords)

def detect_query_type(query):
    try:
        prompt = NON_TDS_DETECTION_PROMPT.format(query=query)
        chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
        resp = chain.invoke({"prompt": prompt}).strip().upper()
        return resp if resp in ["TDS","CONVERSATIONAL","UNRELATED"] else "TDS"
    except:
        return "TDS"

def format_conversation_history(messages):
    history = []
    for msg in messages[-6:]:
        role = "User" if msg["role"]=="user" else "Assistant"
        history.append(f"{role}: {msg['content'][:300]}...")
    return "\n".join(history)

def enhance_image_for_ocr(image):
    """Advanced preprocessing using OpenCV and PIL: grayscale, adaptive threshold, denoise, deskew, resize, contrast, sharpness"""
    try:
        # Convert PIL image to OpenCV format (numpy array)
        img = np.array(image)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Resize if too small
        h, w = img.shape[:2]
        if h < 800 or w < 800:
            scale = max(1.5, 800 / min(h, w))
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        # Adaptive thresholding
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        # Denoise
        img = cv2.medianBlur(img, 3)
        # Deskew
        coords = np.column_stack(np.where(img > 0))
        angle = 0
        if coords.shape[0] > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # Convert back to PIL
        pil_img = Image.fromarray(img)
        # Enhance contrast and sharpness
        pil_img = ImageEnhance.Contrast(pil_img).enhance(2.0)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
        return pil_img
    except Exception as e:
        return image


def extract_invoice_text(uploaded_file):
    """
    Advanced OCR pipeline for invoices:
    - Preprocess with OpenCV (grayscale, threshold, denoise, deskew, resize)
    - Extract full text with Tesseract
    - Detect table-like regions (billed items/services) and extract their text as 'goods/service type'
    - Return (full_text, file_format, confidence, goods_services_list)
    """
    try:
        ftype = uploaded_file.type
        full_text = ""
        confidence = None
        file_format = 'unknown'
        goods_services_list = []
        if ftype.startswith('image'):
            image = Image.open(uploaded_file)
            img = np.array(image)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Resize if too small
            h, w = img.shape[:2]
            if h < 800 or w < 800:
                scale = max(1.5, 800 / min(h, w))
                img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
            # Adaptive thresholding
            img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
            # Denoise
            img_bin = cv2.medianBlur(img_bin, 3)
            # Deskew
            coords = np.column_stack(np.where(img_bin > 0))
            angle = 0
            if coords.shape[0] > 0:
                rect = cv2.minAreaRect(coords)
                angle = rect[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                (h, w) = img_bin.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                img_bin = cv2.warpAffine(img_bin, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            # Extract full text
            pil_img = Image.fromarray(img_bin)
            best_text = ""
            best_conf = -1
            for psm in [6, 3, 4, 7, 8]:
                custom_config = f'--oem 3 --psm {psm}'
                try:
                    data = pytesseract.image_to_data(pil_img, config=custom_config, output_type=pytesseract.Output.DICT)
                    text = " ".join([w for w in data['text'] if w.strip()])
                    conf = (np.mean([float(c) for c in data['conf'] if c != '-1'])
                            if 'conf' in data and len(data['conf']) > 0 else 0)
                    if len(text.strip()) > len(best_text.strip()) or (len(text.strip()) == len(best_text.strip()) and conf > best_conf):
                        best_text = text
                        best_conf = conf
                except Exception:
                    continue
            full_text = best_text
            confidence = best_conf
            file_format = 'image'
            # --- Table detection for goods/services ---
            # Invert for table detection
            img_inv = 255 - img_bin
            # Detect horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
            detect_horizontal = cv2.morphologyEx(img_inv, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            # Detect vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
            detect_vertical = cv2.morphologyEx(img_inv, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            # Combine lines
            table_mask = cv2.add(detect_horizontal, detect_vertical)
            # Find contours (table cells/boxes)
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Extract text from each detected table cell
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 60 and h > 20:  # filter out small boxes
                    roi = img_bin[y:y+h, x:x+w]
                    roi_pil = Image.fromarray(roi)
                    cell_text = pytesseract.image_to_string(roi_pil, config='--oem 3 --psm 6')
                    cell_text = cell_text.strip()
                    if cell_text and len(cell_text) > 2:
                        goods_services_list.append(cell_text)
        # TODO: Add similar logic for PDFs (convert each page to image, repeat above)
        elif ftype == 'application/pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.read()); tmp.flush()
                doc = fitz.open(tmp.name)
                text = ""
                goods_services_list = []
                for page in doc:
                    page_text = page.get_text()
                    if not page_text.strip():
                        pix = page.get_pixmap(dpi=300)
                        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                        # Convert to grayscale for OpenCV
                        img_cv = np.array(img.convert('L'))
                        # Use the same OpenCV pipeline as above
                        # (copy-paste the image pipeline here for brevity)
                        # ...
                        # For now, just extract text
                        ocr_text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')
                        page_text = ocr_text
                    text += page_text + "\n"
                doc.close()
                full_text = text
                file_format = 'pdf'
        elif ftype in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document','application/msword']:
            with tempfile.NamedTemporaryFile(delete=False,suffix='.docx') as tmp:
                tmp.write(uploaded_file.read()); tmp.flush()
                doc = docx.Document(tmp.name)
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            full_text = text
            file_format = 'docx'
        elif ftype=='text/plain':
            full_text = uploaded_file.read().decode(errors="ignore")
            file_format = 'txt'
        else:
            full_text = "[Unsupported file type]"
            file_format = 'unknown'
        return full_text, file_format, confidence, goods_services_list
    except Exception as e:
        return f"[Extraction Error: {e}]", 'unknown', None, []

def extract_invoice_details_with_llm(invoice_text):
    """Simplified invoice extraction - NO TDS classification, only factual data"""
    try:
        prompt = INVOICE_EXTRACTION_PROMPT.format(invoice_text=invoice_text[:4000])
        chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
        response = chain.invoke({"prompt": prompt})
        
        # Try to parse JSON response
        try:
            # Clean up response if it has markdown formatting
            cleaned_response = response.replace('```json', '').replace('```', '').strip()
            invoice_data = json.loads(cleaned_response)
            return invoice_data
        except json.JSONDecodeError:
            # If JSON parsing fails, return a structured format anyway
            return {
                "state": "Not found",
                "city": "Not found", 
                "amount": "Not found",
                "payment_type": "General payment",
                "vendor_name": "Not found",
                "invoice_number": "Not found",
                "date": "Not found",
                "confidence": 50
            }
    except Exception as e:
        return {
            "state": "Error in extraction",
            "city": "Error in extraction", 
            "amount": "Error in extraction",
            "payment_type": "Error in extraction",
            "vendor_name": "Error in extraction",
            "invoice_number": "Error in extraction",
            "date": "Error in extraction",
            "confidence": 0
        }

def format_invoice_info(invoice_data):
    """Format invoice information for LLM consumption"""
    if not invoice_data:
        return "No invoice information available."
    
    info = f"""
**Invoice/Document Information:**
- Vendor: {invoice_data.get('vendor_name', 'Not available')}
- Invoice Number: {invoice_data.get('invoice_number', 'Not available')}
- Date: {invoice_data.get('date', 'Not available')}
- State: {invoice_data.get('state', 'Not available')}
- City: {invoice_data.get('city', 'Not available')}
- Amount: {invoice_data.get('amount', 'Not available')}
- Payment Type/Services: {invoice_data.get('payment_type', 'Not available')}
- Extraction Confidence: {invoice_data.get('confidence', 'Not available')}%
"""
    return info

def format_tds_table(json_response):
    """Convert JSON response to formatted table - NEVER show raw JSON"""
    try:
        # Parse JSON if it's a string
        if isinstance(json_response, str):
            # Try to extract JSON from response if it contains other text
            json_match = re.search(r'\{.*\}', json_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = json_response
            data = json.loads(json_str)
        else:
            data = json_response
        
        results = data.get("analysis_results", [])
        if not results:
            return "## ❌ Analysis Error\n\nNo analysis results found. Please try rephrasing your query or provide more specific information about the transaction."
        
        # Create formatted table
        table = "## 📊 TDS Analysis Results\n\n"
        
        for i, res in enumerate(results, 1):
            table += f"### 📋 Analysis #{i}\n\n"
            table += "| **Aspect** | **Details** |\n"
            table += "|------------|-------------|\n"
            table += f"| **🛍️ Goods/Services** | {res.get('goods_or_services', 'N/A')} |\n"
            table += f"| **📝 Description** | {res.get('explanation', 'N/A')} |\n"
            table += f"| **⚖️ Primary TDS Section** | **{res.get('primary_section', 'N/A')}** |\n"
            table += f"| **💰 Primary TDS Rate** | **{res.get('primary_rate', 'N/A')}** |\n"
            table += f"| **📖 Primary Justification** | {res.get('primary_explanation', 'N/A')} |\n"
            table += f"| **🔄 Alternate TDS Section** | {res.get('alternate_section', 'None')} |\n"
            table += f"| **💸 Alternate TDS Rate** | {res.get('alternate_rate', 'N/A')} |\n"
            table += f"| **📚 Alternate Justification** | {res.get('alternate_explanation', 'N/A')} |\n"
            # Removed confidence score row from table
            table += "\n"
        
        # Add summary information
        table += "---\n\n"
        table += f"**🎲 Overall Confidence:** {data.get('overall_confidence', 'N/A')}%\n\n"
        table += f"**📚 Sources:** {', '.join(data.get('sources', ['N/A']))}\n\n"
        
        if data.get('additional_notes'):
            table += f"**📌 Additional Notes:** {data.get('additional_notes')}\n\n"
        
        table += "---\n\n"
        table += "💡 **Need more information?** Feel free to ask follow-up questions about TDS compliance, exemptions, or specific scenarios!"
        
        return table
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails, create a formatted response from the raw text
        return f"""## 📊 TDS Analysis Results

**Response:** {json_response}

---

💡 **Note:** The response couldn't be formatted as a table. Please try rephrasing your query for better structured results.

**Need more information?** Feel free to ask follow-up questions about TDS compliance, exemptions, or specific scenarios!"""
        
    except Exception as e:
        return f"""## ❌ Analysis Error

There was an error processing the TDS analysis: {str(e)}

**Raw Response for Reference:**
{json_response}

---

💡 **Suggestion:** Please try rephrasing your query or provide more specific information about the transaction.

**Need help?** Ask about specific TDS sections, rates, or compliance requirements!"""

def run_tds_analysis(query, conversation_history, invoice_data=None):
    # 1️⃣ Detect query type
    qtype = detect_query_type(query)
    if qtype=="UNRELATED":
        return "I don't have enough knowledge to answer that question. However, I'm here to help with any TDS-related queries you might have!"
    if qtype=="CONVERSATIONAL":
        greetings = [
            "Hello! I'm your TDS expert assistant.",
            "Hi there! How can I help you with TDS matters today?",
            "Great! I'm here to help you with any TDS-related questions.",
            "Good to see you! What TDS query can I assist with?"
        ]
        return f"{greetings[len(conversation_history)%len(greetings)]} Feel free to ask me about TDS rates, sections, compliance, or upload documents for analysis."

    # 2️⃣ RAG context from vector database
    context = get_context_for_query(query)

    # 3️⃣ Format invoice information
    invoice_info = format_invoice_info(invoice_data)

    # 4️⃣ Enhance query with invoice payment type if available
    enhanced_query = query
    if invoice_data and invoice_data.get('payment_type') and invoice_data['payment_type'] != "Not found":
        enhanced_query = f"[Payment Type from Invoice: {invoice_data['payment_type']}] {query}"

    # 5️⃣ Choose prompt and ensure proper formatting
    if is_tds_rate_section_query(enhanced_query):
        # TDS Rate/Section queries - Use vector database for classification
        prompt = TDS_ANALYSIS_PROMPT.format(
            query=enhanced_query, context=context, invoice_info=invoice_info
        )
        response = (ChatPromptTemplate.from_template("{prompt}") 
                    | llm | StrOutputParser()).invoke({"prompt": prompt})
        # ALWAYS format as table, never show raw JSON
        return format_tds_table(response)
    else:
        # General TDS queries - ALWAYS return paragraph format
        prompt = GENERAL_TDS_PROMPT.format(
            context=context,
            invoice_info=invoice_info,
            conversation_history=format_conversation_history(conversation_history),
            query=enhanced_query
        )
        response = (ChatPromptTemplate.from_template("{prompt}")
                | llm | StrOutputParser()).invoke({"prompt": prompt})
        # Ensure it's in paragraph format (not JSON or table)
        return response

# Streamlit UI
st.set_page_config(page_title="TDS Chatbot - Expert Assistant", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "invoice_data" not in st.session_state:
    st.session_state.invoice_data = None
if "processing_upload" not in st.session_state:
    st.session_state.processing_upload = False

st.title("🤖 TDS Expert Chatbot")
st.markdown("Ask me anything about TDS provisions, rates, sections, or compliance!")

# File upload section integrated into the main chat interface
if not st.session_state.invoice_data and not st.session_state.processing_upload:
    with st.container():
        st.markdown("### 📄 Upload Invoice/Document (Optional)")
        uploaded_file = st.file_uploader(
            "Upload a document to extract payment details for TDS analysis",
            type=["pdf","txt","docx","png","jpg","jpeg"],
            key="main_file_uploader"
        )
        if uploaded_file is not None:
            st.session_state.processing_upload = True
            with st.spinner("🔍 Extracting payment details from document..."):
                full_text, file_format, ocr_conf, goods_services_list = extract_invoice_text(uploaded_file)
                invoice_data = extract_invoice_details_with_llm(full_text)
                st.session_state.invoice_data = invoice_data
                st.session_state.extracted_text = full_text
                st.session_state.ocr_conf = ocr_conf
                st.session_state.file_format = file_format
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.goods_services_list = goods_services_list
                st.success("✅ Document processed successfully!")
                # Always show uploaded file preview or name
                if file_format == 'image':
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Document", width=400)
                else:
                    st.write(f"**Uploaded File:** {uploaded_file.name}")
                # Always show extracted text
                st.markdown("#### 📝 Extracted Text")
                st.text_area("Extracted Content", full_text, height=300, disabled=True)
                st.download_button("Download Extracted Text", full_text, file_name="extracted_text.txt")
                # Show detected goods/services
                if goods_services_list:
                    st.markdown("#### 📦 Detected Goods/Service Types (from table)")
                    for idx, item in enumerate(goods_services_list, 1):
                        st.write(f"{idx}. {item}")
                # Warning if text is short or confidence is low
                if (ocr_conf is not None and ocr_conf < 60) or len(full_text.strip()) < 50:
                    st.warning(f"⚠️ OCR confidence is low ({ocr_conf:.1f}%). The extracted text may be incomplete or inaccurate.")
                st.info("💡 **Next Step:** Ask me about TDS rates or sections for these services, and I'll analyze using the legal knowledge base!")
            st.session_state.processing_upload = False
            st.rerun()

# Display current invoice context if available
if st.session_state.invoice_data:
    with st.container():
        st.markdown("### 📄 Current Document Context")
        col1, col2, col3 = st.columns([2,2,1])
        
        with col1:
            st.write(f"**Services/Goods:** {st.session_state.invoice_data.get('payment_type', 'Not available')}")
            st.write(f"**Vendor:** {st.session_state.invoice_data.get('vendor_name', 'Not available')}")
        
        with col2:
            st.write(f"**Amount:** {st.session_state.invoice_data.get('amount', 'Not available')}")
            st.write(f"**Location:** {st.session_state.invoice_data.get('city', 'Not available')}, {st.session_state.invoice_data.get('state', 'Not available')}")
        
        with col3:
            if st.button("🗑️ Clear Document"):
                st.session_state.invoice_data = None
                st.session_state.extracted_text = None
                st.session_state.goods_services_list = None
                st.rerun()
        st.markdown("---")
        # Always show extracted text if available
        if st.session_state.get('extracted_text'):
            st.markdown("#### 📝 Extracted Text")
            st.text_area("Extracted Content", st.session_state.extracted_text, height=300, disabled=True)
            st.download_button("Download Extracted Text", st.session_state.extracted_text, file_name="extracted_text.txt")
        # Show detected goods/services if available
        if st.session_state.get('goods_services_list'):
            st.markdown("#### 📦 Detected Goods/Service Types (from table)")
            for idx, item in enumerate(st.session_state.goods_services_list, 1):
                st.write(f"{idx}. {item}")

# Chat interface with scrollable container
chat_container = st.container()

with chat_container:
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask your TDS question..."):
        # Add user message
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your query using TDS knowledge base..."):
                response = run_tds_analysis(
                    prompt,
                    st.session_state.messages,
                    st.session_state.invoice_data
                )
            st.markdown(response)
        
        # Add assistant response
        st.session_state.messages.append({"role":"assistant","content":response})

# Quick start section (only show if no messages)
if not st.session_state.messages:
    st.markdown("### 🚀 Quick Start")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 TDS Rates Overview"):
            st.session_state.messages.append({"role":"user","content":"What are the different TDS rates for various types of payments?"})
            st.rerun()
    
    with col2:
        if st.button("📋 TDS Compliance"):
            st.session_state.messages.append({"role":"user","content":"What are the key TDS compliance requirements I should know?"})
            st.rerun()
    
    with col3:
        if st.button("🔍 Section 194C Analysis"):
            st.session_state.messages.append({"role":"user","content":"Analyze TDS section and rate for contractor payments"})
            st.rerun()

# Sidebar with additional controls and information
with st.sidebar:
    st.header("🛠️ Chat Controls")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    st.subheader("📋 Current Session Info")
    st.write(f"**Messages:** {len(st.session_state.messages)}")
    st.write(f"**Document Uploaded:** {'✅ Yes' if st.session_state.invoice_data else '❌ No'}")
    
    if st.session_state.invoice_data:
        st.write(f"**Services/Goods:** {st.session_state.invoice_data.get('payment_type', 'N/A')[:30]}...")
        st.write(f"**Confidence:** {st.session_state.invoice_data.get('confidence', 0)}%")
    
    st.markdown("---")
    
    st.subheader("🔄 Workflow")
    st.markdown("""
    **Enhanced Two-Step Process:**
    
    1. **📄 Document Upload & Extraction**
       - OCR extracts factual data only
       - No TDS classification at this stage
       - Focus on payment type, amount, location
    
    2. **🎯 TDS Analysis via Query**
       - Vector database provides legal context
       - LLM analyzes payment type against TDS rules
       - Accurate section & rate determination
    """)
    
    st.markdown("---")
    
    st.subheader("💡 Key Features")
    st.markdown("""
    - **🔍 Smart OCR**: Extracts payment details without premature TDS classification
    - **📊 Vector-based Analysis**: Uses legal knowledge base for accurate TDS classification
    - **🧠 Context Retention**: Remembers invoice details throughout conversation
    - **💬 Natural Chat**: Conversational interface with structured analysis
    - **📄 Multi-format Support**: PDF, DOCX, images, and text files
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666;'>"
    "💼 Simplified TDS Expert Assistant - OCR + Vector DB Analysis | "
    "📚 Based on Income Tax Act 1961 - TDS Provisions 2024"
    "</div>",
    unsafe_allow_html=True
)