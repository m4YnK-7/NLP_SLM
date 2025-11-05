import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import re

# Page config
st.set_page_config(
    page_title="Fee Receipt Generator",
    page_icon="🧾",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .receipt-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .field-label {
        font-weight: bold;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the fine-tuned model"""
    model_path = "./models/flan_t5_fee_extractor"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure you've trained the model first using train_model.py")
        return None, None, None

def extract_payment_details(message, tokenizer, model, device):
    """Extract payment details using the fine-tuned model"""
    # Format input like training data
    input_text = f"Extract payment details: {message}"
    
    # Tokenize and generate
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode output
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse the output
    details = {}
    parts = decoded.split('|')
    for part in parts:
        if ':' in part:
            key, value = part.split(':', 1)
            details[key.strip()] = value.strip()
    
    return details

def create_pdf_receipt(details):
    """Generate PDF receipt from extracted details"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    # Title
    title = Paragraph("OFFICIAL FEE RECEIPT", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.3*inch))
    
    # Institute name
    institute = details.get('institute', 'Institute Name')
    institute_para = Paragraph(f"<b>{institute}</b>", styles['Heading2'])
    elements.append(institute_para)
    elements.append(Spacer(1, 0.2*inch))
    
    # Receipt details table
    receipt_data = [
        ['Receipt No:', details.get('transaction_id', 'N/A')],
        ['Date:', details.get('date', 'N/A')],
        ['', ''],
        ['Student Name:', details.get('student_name', 'N/A')],
        ['Roll Number:', details.get('roll_number', 'N/A')],
        ['', ''],
        ['Payment For:', details.get('payment_name', 'N/A')],
        ['Amount Paid:', f"₹{details.get('amount', '0')}"],
    ]
    
    table = Table(receipt_data, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 12),
        ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 12),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#333333')),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f0f2f6')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#1f77b4')),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 0.5*inch))
    
    # Footer
    footer_text = f"<i>Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</i>"
    footer = Paragraph(footer_text, styles['Italic'])
    elements.append(footer)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Main app
def main():
    st.markdown('<div class="main-header">🧾 Fee Receipt Generator</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading AI model..."):
        tokenizer, model, device = load_model()
    
    if tokenizer is None:
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    st.info("📱 Paste your payment SMS/Email below to generate an official receipt")
    
    # Input section
    st.subheader("📥 Input Payment Message")
    
    # Sample messages for testing
    sample_messages = [
        "Payment of ₹15000 received from Priya Sharma (Roll No: R2451) for Tuition Fee on March 15, 2025. Txn ID: TXN8932ABCD5621. - SkillUp Academy",
        "Hi Nathan Martinez, we've received ₹8000 towards Lab Fee on August 20, 2025. Ref: TXN5621XYZW9876, Roll: R1894. - Bright Minds Academy",
        "Dear Emily Brown, your payment of ₹20000 for Hostel Rent has been confirmed. Transaction ID: TXN7734PQRS4455, Date: June 10, 2025, Roll Number: R3567. - Excellence Institute"
    ]
    
    # Sample selector
    use_sample = st.checkbox("📋 Use sample message")
    if use_sample:
        selected_sample = st.selectbox("Choose a sample:", range(len(sample_messages)), format_func=lambda x: f"Sample {x+1}")
        message = st.text_area("Payment Message:", sample_messages[selected_sample], height=150)
    else:
        message = st.text_area("Payment Message:", height=150, placeholder="Paste your payment confirmation SMS/Email here...")
    
    # Extract button
    if st.button("🔍 Extract Details & Generate Receipt", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("⚠️ Please enter a payment message")
            return
        
        with st.spinner("🤖 AI is extracting payment details..."):
            try:
                details = extract_payment_details(message, tokenizer, model, device)
                
                if not details:
                    st.error("❌ Could not extract details from the message")
                    return
                
                # Display extracted details
                st.markdown("---")
                st.subheader("✅ Extracted Details")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**👤 Student Name:** {details.get('student_name', 'N/A')}")
                    st.markdown(f"**🎓 Roll Number:** {details.get('roll_number', 'N/A')}")
                    st.markdown(f"**💰 Amount:** ₹{details.get('amount', 'N/A')}")
                
                with col2:
                    st.markdown(f"**📅 Date:** {details.get('date', 'N/A')}")
                    st.markdown(f"**🔢 Transaction ID:** {details.get('transaction_id', 'N/A')}")
                    st.markdown(f"**📝 Payment Type:** {details.get('payment_name', 'N/A')}")
                
                st.markdown(f"**🏫 Institute:** {details.get('institute', 'N/A')}")
                
                # Generate PDF
                st.markdown("---")
                st.subheader("📄 Official Receipt")
                
                with st.spinner("Generating PDF receipt..."):
                    pdf_buffer = create_pdf_receipt(details)
                
                # Download button
                st.download_button(
                    label="📥 Download Receipt (PDF)",
                    data=pdf_buffer,
                    file_name=f"Receipt_{details.get('transaction_id', 'unknown')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
                
                st.success("✅ Receipt generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("💡 Tip: Make sure your message contains student name, amount, roll number, date, transaction ID, payment type, and institute name")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>🤖 Powered by FLAN-T5 Fine-tuned Model</p>
        <p>Built with Streamlit • ReportLab</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()