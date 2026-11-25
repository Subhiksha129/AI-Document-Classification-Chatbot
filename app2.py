import streamlit as st
import joblib
import sqlite3
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pypdf import PdfReader
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="AI Document Classification",
    page_icon="📄",
    layout="wide"
)

st.title("AI Document Classification Chatbot")
st.write("Upload PDF documents, classify them automatically, ask questions about uploaded PDFs, and monitor classification activity.")

MODEL_DIR = Path("models")
DB_PATH = "document_history.db"

# ---------------- DATABASE ----------------

def create_database():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS document_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            upload_date TEXT,
            main_categories TEXT,
            detailed_categories TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS deleted_document_history (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            upload_date TEXT,
            main_categories TEXT,
            detailed_categories TEXT,
            deleted_date TEXT
        )
    """)

    conn.commit()
    conn.close()



def save_document(filename, main_categories, detail_categories):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO document_history
        (filename, upload_date, main_categories, detailed_categories)
        VALUES (?, ?, ?, ?)
    """, (
        filename,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        json.dumps(list(main_categories)),
        json.dumps(list(detail_categories))
    ))
    conn.commit()
    conn.close()


def delete_document(document_id):
    conn = sqlite3.connect(DB_PATH)

    record = conn.execute(
        "SELECT * FROM document_history WHERE id = ?",
        (document_id,)
    ).fetchone()

    if record:
        conn.execute("""
            INSERT INTO deleted_document_history
            (id, filename, upload_date, main_categories,
             detailed_categories, deleted_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            record[0],
            record[1],
            record[2],
            record[3],
            record[4],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.execute(
            "DELETE FROM document_history WHERE id = ?",
            (document_id,)
        )

    conn.commit()
    conn.close()

def restore_document(document_id):
    conn = sqlite3.connect(DB_PATH)

    record = conn.execute(
        "SELECT * FROM deleted_document_history WHERE id = ?",
        (document_id,)
    ).fetchone()

    if record:
        conn.execute("""
            INSERT INTO document_history
            (id, filename, upload_date, main_categories, detailed_categories)
            VALUES (?, ?, ?, ?, ?)
        """, (
            record[0],
            record[1],
            record[2],
            record[3],
            record[4]
        ))

        conn.execute(
            "DELETE FROM deleted_document_history WHERE id = ?",
            (document_id,)
        )

    conn.commit()
    conn.close()

def clear_deleted_history():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("DELETE FROM deleted_document_history")

    conn.commit()
    conn.close()





def get_document_history():
    conn = sqlite3.connect(DB_PATH)
    history = pd.read_sql_query(
        "SELECT * FROM document_history ORDER BY id DESC", conn
    )
    conn.close()
    return history

def get_deleted_document_history():
    conn = sqlite3.connect(DB_PATH)

    history = pd.read_sql_query(
        "SELECT * FROM deleted_document_history ORDER BY deleted_date DESC",
        conn
    )

    conn.close()
    return history



def get_category_counts(data, column):
    counts = {}
    for value in data[column]:
        try:
            categories = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            categories = []
        for category in categories:
            counts[category] = counts.get(category, 0) + 1
    return counts


create_database()

# ---------------- LOAD CLASSIFICATION MODELS ----------------
@st.cache_resource

def load_models():
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer_improved.pkl")
    main_model = joblib.load(MODEL_DIR / "main_category_model_improved.pkl")
    main_encoder = joblib.load(MODEL_DIR / "main_label_encoder_improved.pkl")
    detail_model = joblib.load(MODEL_DIR / "detail_category_model_improved.pkl")
    detail_encoder = joblib.load(MODEL_DIR / "detail_label_encoder_improved.pkl")

    return vectorizer, main_model, main_encoder, detail_model, detail_encoder


# ---------------- PDF EXTRACTION ----------------
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    extracted_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_text += page_text + "\n"
    return extracted_text
# ---------------- CLASSIFICATION FALLBACK ----------------
def predict_categories_with_fallback(model, encoder, features, top_k):
    prediction = model.predict(features)
    results = list(encoder.inverse_transform(prediction)[0])

    if results:
        return results, False

    try:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(features)[0]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(features)[0]
        else:
            return [], False

        labels = list(encoder.classes_)

        if len(scores) != len(labels):
            return [], False

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: float(scores[i]),
            reverse=True
        )

        best_indices = ranked_indices[:min(top_k, len(ranked_indices))]

        suggested_categories = [
            str(labels[i]) for i in best_indices
        ]

        return suggested_categories, True

    except Exception:
        return [], False

#
def classify_academic_document(text):
    text = text.lower()

    academic_categories = {
        "Data Structures and Algorithms": [
            "algorithm", "data structure", "divide and conquer",
            "merge sort", "quick sort", "dynamic programming",
            "binary search", "time complexity", "space complexity"
        ],

        "Database Management": [
            "database", "sql", "mysql", "normalization",
            "primary key", "foreign key", "dbms"
        ],

        "Operating Systems": [
            "operating system", "process management",
            "deadlock", "memory management", "cpu scheduling",
            "process scheduling", "system calls", "file system",
            "virtual memory", "multiprogramming", "multitasking",
            "kernel mode", "user mode", "booting"
        ],

        "Computer Networks": [
            "computer network", "tcp", "ip address",
            "osi model", "network protocol"
        ],

        "Cybersecurity": [
            "cybersecurity", "encryption", "firewall",
            "malware", "cryptography", "cyber attack"
        ],

        "Programming": [
            "python", "java", "c++", "programming language"
        ],

        "Artificial Intelligence": [
            "artificial intelligence", "neural network",
            "deep learning", "intelligent agent"
        ],

        "Machine Learning": [
            "machine learning", "supervised learning",
            "unsupervised learning", "regression",
            "classification algorithm", "training data"
        ],

        "Natural Language Processing": [
            "natural language processing", "nlp",
            "text processing", "tokenization",
            "lemmatization", "stemming", "sentiment analysis"
        ],

        "Mathematics": [
            "linear algebra", "calculus", "probability",
            "statistics", "matrix", "differentiation",
            "integration", "trigonometry", "algebra"
        ],

        "Software Engineering": [
            "software engineering", "software development",
            "software testing", "software design",
            "software requirements", "agile methodology"
        ],

        "Computer Organization": [
            "computer organization", "computer architecture",
            "instruction set", "register", "memory hierarchy",
            "mips", "addressing mode", "control unit",
            "arithmetic logic unit", "pipelining"
        ],

        "Computer Science": [
            "computational theory", "theory of computation",
            "finite automata", "turing machine",
            "compiler design"
        ],

        "Physics": [
            "physics", "motion", "force", "energy",
            "newton's laws", "velocity", "acceleration",
            "gravity", "thermodynamics", "electromagnetism",
            "optics", "wave motion"
        ],

        "Chemistry": [
            "chemistry", "chemical reaction", "periodic table",
            "atomic structure", "molecule", "chemical bonding",
            "organic chemistry", "inorganic chemistry",
            "acid", "base", "oxidation"
        ],

        "Biology": [
            "biology", "cell structure", "photosynthesis",
            "respiration", "genetics", "dna", "rna",
            "evolution", "ecosystem", "microorganism",
            "human anatomy"
        ],

        "Electronics": [
            "electronics", "semiconductor", "diode",
            "transistor", "amplifier", "logic gates",
            "digital circuits", "analog circuits",
            "integrated circuit", "microprocessor"
        ],

        "Electrical Engineering": [
            "electrical engineering", "electric circuit",
            "transformer", "electric motor", "generator",
            "power system", "alternating current",
            "direct current", "electrical machine"
        ],

        "Mechanical Engineering": [
            "mechanical engineering", "thermodynamics",
            "fluid mechanics", "machine design",
            "manufacturing process", "heat transfer",
            "kinematics", "dynamics"
        ],

        "Civil Engineering": [
            "civil engineering", "structural engineering",
            "concrete", "reinforced cement concrete",
            "soil mechanics", "surveying",
            "building materials", "foundation engineering"
        ]
    }

    scores = {}

    for category, keywords in academic_categories.items():
        matched_keywords = [
            keyword for keyword in keywords
            if keyword in text
        ]

        scores[category] = len(matched_keywords)

    matched_categories = [
        max(scores, key=scores.get)
    ] if max(scores.values()) >= 2 else []

    if matched_categories:
        return "Education", matched_categories

    return None, []


# ---------------- CHATBOT MODELS ----------------
@st.cache_resource
def load_chatbot_models():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    return embeddings, llm


# Build a temporary vector store only from the PDFs selected in this chatbot session.
def create_document_vectorstore(uploaded_files):
    documents = []

    for pdf_file in uploaded_files:
        pdf_file.seek(0)
        reader = PdfReader(pdf_file)

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()

            if page_text and page_text.strip():
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": pdf_file.name,
                            "page": page_number
                        }
                    )
                )

    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings, llm = load_chatbot_models()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore, llm


def answer_document_question(vectorstore, llm, question):
    relevant_chunks = vectorstore.similarity_search(question, k=4)

    if not relevant_chunks:
        return (
            "The uploaded documents do not contain information relevant to this question.",
            []
        )

    context_parts = []
    sources = []
    seen_sources = set()

    for chunk in relevant_chunks:
        source = chunk.metadata.get("source", "Unknown PDF")
        page = chunk.metadata.get("page", "Unknown")

        context_parts.append(
            f"[Source: {source}, Page: {page}]\n{chunk.page_content}"
        )

        source_key = (source, page)

        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "source": source,
                "page": page
            })

    context = "\n\n".join(context_parts)

    prompt = f"""You are a PDF question-answering assistant.
Use ONLY the context below to answer.

If the answer is not clearly contained in the context, reply exactly:
"The uploaded documents do not contain this information."

Do not use outside knowledge or make up details.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)

    return response.content, sources

# ---------------- REPORT CHART IMAGE ----------------
def create_chart_image(data, title, x_label, y_label):
    fig, ax = plt.subplots(figsize=(8, 4))
    data.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    image_buffer = BytesIO()
    fig.savefig(image_buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    image_buffer.seek(0)
    return image_buffer


# ---------------- WEEKLY PDF REPORT ----------------
def generate_weekly_report(history):
    now = datetime.now()
    week_start = (now - timedelta(days=now.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    history = history.copy()
    history["upload_date"] = pd.to_datetime(history["upload_date"], errors="coerce")
    weekly_history = history[
        (history["upload_date"] >= week_start) &
        (history["upload_date"] <= now)
    ].copy()

    if weekly_history.empty:
        return None

    main_counts = get_category_counts(weekly_history, "main_categories")
    detail_counts = get_category_counts(weekly_history, "detailed_categories")
    pdf_buffer = BytesIO()
    pdf_doc = SimpleDocTemplate(
        pdf_buffer, pagesize=A4,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI Document Classification - Weekly Report", styles["Title"]))
    story.append(Spacer(1, 15))
    story.append(Paragraph(
        f"Report Period: {week_start.strftime('%d-%m-%Y')} to {now.strftime('%d-%m-%Y %H:%M')}",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 15))
    story.append(Paragraph("Weekly Summary", styles["Heading2"]))

    summary_data = [
        ["Metric", "Value"],
        ["Total Documents Classified", str(len(weekly_history))],
        ["Main Categories Found", str(len(main_counts))],
        ["Detailed Categories Found", str(len(detail_counts))]
    ]
    summary_table = Table(summary_data, colWidths=[300, 150])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE8DF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))

    if main_counts:
        story.append(Paragraph("Main Category Distribution", styles["Heading2"]))
        main_series = pd.Series(main_counts).sort_values(ascending=False)
        main_chart = create_chart_image(main_series, "Main Category Distribution", "Category", "Documents")
        story.append(Image(main_chart, width=6.5 * inch, height=3.2 * inch))
        story.append(Spacer(1, 15))

    if detail_counts:
        story.append(Paragraph("Detailed Category Distribution", styles["Heading2"]))
        detail_series = pd.Series(detail_counts).sort_values(ascending=False).head(10)
        detail_chart = create_chart_image(detail_series, "Top 10 Detailed Categories", "Category", "Documents")
        story.append(Image(detail_chart, width=6.5 * inch, height=3.2 * inch))
        story.append(Spacer(1, 15))

    story.append(Paragraph("Document Activity This Week", styles["Heading2"]))
    weekly_history["Date"] = weekly_history["upload_date"].dt.strftime("%d-%m")
    activity = weekly_history.groupby("Date").size()
    activity_chart = create_chart_image(activity, "Documents Classified This Week", "Date", "Documents")
    story.append(Image(activity_chart, width=6.5 * inch, height=3.2 * inch))
    story.append(PageBreak())

    story.append(Paragraph("Classified Documents", styles["Heading2"]))
    document_data = [["Filename", "Date", "Main Categories", "Detailed Categories"]]
    for _, record in weekly_history.iterrows():
        try:
            main_labels = ", ".join(json.loads(record["main_categories"]))
        except (TypeError, json.JSONDecodeError):
            main_labels = ""
        try:
            detail_labels = ", ".join(json.loads(record["detailed_categories"]))
        except (TypeError, json.JSONDecodeError):
            detail_labels = ""
        document_data.append([
            Paragraph(str(record["filename"]), styles["BodyText"]),
            record["upload_date"].strftime("%d-%m-%Y"),
            Paragraph(main_labels, styles["BodyText"]),
            Paragraph(detail_labels, styles["BodyText"])
        ])

    document_table = Table(document_data, colWidths=[115, 65, 145, 145], repeatRows=1)
    document_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE8DF")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
    ]))
    story.append(document_table)
    pdf_doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# ---------------- LOAD TRAINED MODELS ----------------
try:
    (
        vectorizer, main_model, main_encoder,
        detail_model, detail_encoder
    ) = load_models()
    st.success("AI classification models loaded successfully.")
except Exception as e:
    st.error(f"Unable to load models: {e}")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Document Classification", "Dashboard", "Document History", "Document Chatbot"]
)


# ---------------- DOCUMENT CLASSIFICATION ----------------
if page == "Document Classification":
    st.header("Upload Your Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="classification_pdf_upload"
    )

    if uploaded_files:
        st.write(f"Selected documents: {len(uploaded_files)}")

        for pdf in uploaded_files:
            st.write("📄", pdf.name)

        if st.button("Classify All Documents"):

            st.session_state["classification_records"] = []
            st.session_state["show_csv_report"] = False

            for uploaded_file in uploaded_files:

                with st.spinner(f"Analysing {uploaded_file.name}..."):

                    try:
                        document_text = extract_pdf_text(uploaded_file)

                        if not document_text.strip():
                            st.session_state["classification_records"].append({
                                "Filename": uploaded_file.name,
                                "Main Category": "Failed",
                                "Detailed Category": "",
                                "Status": "Failed",
                                "Date": datetime.now().strftime("%d-%m-%Y"),
                                "Time": datetime.now().strftime("%H:%M:%S"),
                                "Extracted Words": 0,
                                "Matched TF-IDF Features": 0,
                                "Extracted Text Length": 0
                            })

                            st.error(
                                f"Unable to extract readable text from {uploaded_file.name}. "
                                "The PDF may be empty or corrupted."
                            )
                            continue

                        features = vectorizer.transform([document_text])

                        main_results, main_used_fallback = (
                            predict_categories_with_fallback(
                                main_model,
                                main_encoder,
                                features,
                                top_k=2
                            )
                        )

                        detail_results, detail_used_fallback = (
                            predict_categories_with_fallback(
                                detail_model,
                                detail_encoder,
                                features,
                                top_k=3
                            )
                        )

                        academic_main, academic_details = (
                            classify_academic_document(document_text)
                        )

                        if academic_main:
                            main_results = [academic_main]
                            detail_results = academic_details
                            classification_status = "Classified"
                        else:
                            main_results = ["Unclassified"]
                            detail_results = []
                            classification_status = "Unclassified"

                        save_document(
                            uploaded_file.name,
                            main_results,
                            detail_results
                        )

                        st.session_state["classification_records"].append({
                            "Filename": uploaded_file.name,
                            "Main Category": ", ".join(main_results),
                            "Detailed Category": ", ".join(detail_results),
                            "Status": classification_status,
                            "Date": datetime.now().strftime("%d-%m-%Y"),
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "Extracted Words": len(document_text.split()),
                            "Matched TF-IDF Features": features.nnz,
                            "Extracted Text Length": len(document_text)
                        })

                    except Exception as e:
                        st.session_state["classification_records"].append({
                            "Filename": uploaded_file.name,
                            "Main Category": "",
                            "Detailed Category": "",
                            "Status": "Failed",
                            "Date": datetime.now().strftime("%d-%m-%Y"),
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "Extracted Words": 0,
                            "Matched TF-IDF Features": 0,
                            "Extracted Text Length": 0
                        })

                        st.error(f"Error processing {uploaded_file.name}: {e}")

        # Display saved classification results
        saved_records = st.session_state.get(
            "classification_records", []
        )

        if saved_records:

            st.divider()
            st.subheader("Classification Results")

            for record in saved_records:

                with st.expander(
                    f"Results: {record['Filename']}",
                    expanded=True
                ):
                    st.write("Status:", record["Status"])

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Main Categories")
                        st.write(record["Main Category"])

                    with col2:
                        st.markdown("### Detailed Categories")
                        if record["Detailed Category"]:
                            st.write(record["Detailed Category"])
                        else:
                            st.info("No detailed category available.")

                    st.write("Extracted words:", record["Extracted Words"])
                    st.write(
                        "Matched TF-IDF features:",
                        record["Matched TF-IDF Features"]
                    )
                    st.write(
                        "Extracted text length:",
                        record["Extracted Text Length"],
                        "characters"
                    )

            # Classification Summary
            classified_count = sum(
                1 for r in saved_records if r["Status"] == "Classified"
            )

            unclassified_count = sum(
                1 for r in saved_records if r["Status"] == "Unclassified"
            )

            failed_count = sum(
                1 for r in saved_records if r["Status"] == "Failed"
            )

            st.divider()
            st.subheader("Classification Summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Documents", len(saved_records))
            col2.metric("Classified", classified_count)
            col3.metric("Unclassified", unclassified_count)
            col4.metric("Failed", failed_count)

            # CSV Report
            report_columns = [
                "Filename",
                "Main Category",
                "Detailed Category",
                "Status",
                "Date",
                "Time"
            ]

            report_df = pd.DataFrame(saved_records)[report_columns]

            st.divider()
            st.subheader("CSV Report")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Open CSV Report"):
                    st.session_state["show_csv_report"] = True

            with col2:
                csv_data = report_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download CSV Report",
                    data=csv_data,
                    file_name="classification_report.csv",
                    mime="text/csv"
                )

            if st.session_state.get("show_csv_report", False):
                st.subheader("Classification Report")
                st.dataframe(report_df, use_container_width=True)



# ---------------- DASHBOARD ----------------
elif page == "Dashboard":
    st.header("Classification Dashboard")
    history = get_document_history()

    if history.empty:
        st.info("No classified documents available yet.")
    else:
        total_documents = len(history)
        history["upload_date"] = pd.to_datetime(history["upload_date"], errors="coerce")
        valid_history = history.dropna(subset=["upload_date"]).copy()
        main_counts = get_category_counts(history, "main_categories")
        detail_counts = get_category_counts(history, "detailed_categories")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Documents", total_documents)
        col2.metric("Main Categories Found", len(main_counts))
        col3.metric("Detailed Categories Found", len(detail_counts))
        st.divider()

        st.subheader("Main Category Distribution")
        if main_counts:
            main_df = pd.DataFrame(list(main_counts.items()), columns=["Category", "Documents"]).sort_values("Documents", ascending=False)
            st.bar_chart(main_df.set_index("Category"))

        st.subheader("Detailed Category Distribution")
        if detail_counts:
            detail_df = pd.DataFrame(list(detail_counts.items()), columns=["Category", "Documents"]).sort_values("Documents", ascending=False)
            st.bar_chart(detail_df.head(10).set_index("Category"))
            st.caption("Showing the top 10 detailed categories.")

        st.divider()
        st.subheader("Weekly Document Activity")
        if not valid_history.empty:
            valid_history["Week"] = valid_history["upload_date"].dt.to_period("W").astype(str)
            weekly_counts = valid_history.groupby("Week").size().reset_index(name="Documents")
            st.line_chart(weekly_counts.set_index("Week"))

        
        st.divider()
        st.subheader("Daily Document Activity")

        if not valid_history.empty:
            valid_history["Date"] = valid_history["upload_date"].dt.strftime("%d-%m-%Y")

            daily_counts = (
                valid_history.groupby("Date")
                .size()
                .reset_index(name="Documents")
            )

            st.bar_chart(daily_counts.set_index("Date"))

        st.divider()
        st.subheader("Weekly PDF Report")
        st.write("Generate a downloadable report containing this week's classification summary, charts and document details.")
        if st.button("Generate Weekly PDF Report"):
            with st.spinner("Generating weekly report..."):
                try:
                    report_data = generate_weekly_report(history)
                    if report_data is None:
                        st.session_state.pop("weekly_report_bytes", None)
                        st.warning("No documents have been classified this week. Classify a document first to generate the report.")
                    else:
                        st.session_state["weekly_report_bytes"] = report_data
                        st.session_state["weekly_report_filename"] = f"Weekly_Classification_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
                        st.success("Weekly PDF report generated successfully!")
                except Exception as e:
                    st.error(f"Report generation error: {e}")

        if st.session_state.get("weekly_report_bytes"):
            st.download_button(
                label="Download Weekly PDF Report",
                data=st.session_state["weekly_report_bytes"],
                file_name=st.session_state["weekly_report_filename"],
                mime="application/pdf"
            )

# ---------------- DOCUMENT HISTORY ----------------


elif page == "Document History":
    st.header("Document History")

    history = get_document_history()

    if history.empty:
        st.info("No documents have been classified yet.")

    else:
        search_query = st.text_input(
            "Search Documents",
            placeholder="Enter filename or category..."
        )

        filtered_history = history.copy()

        if search_query:
            search_query = search_query.lower()

            filtered_history = filtered_history[
                filtered_history.apply(
                    lambda record: (
                        search_query in str(record["filename"]).lower()
                        or search_query in str(record["main_categories"]).lower()
                        or search_query in str(record["detailed_categories"]).lower()
                    ),
                    axis=1
                )
            ]

        st.write("Documents found:", len(filtered_history))

        if filtered_history.empty:
            st.warning("No matching documents found.")

        else:
            for _, record in filtered_history.iterrows():
                col1, col2 = st.columns([5, 1])

                with col1:
                    with st.expander(
                        f"{record['filename']} — {record['upload_date']}"
                    ):
                        st.write(
                            "**Main Categories:**",
                            ", ".join(json.loads(record["main_categories"]))
                        )

                        st.write(
                            "**Detailed Categories:**",
                            ", ".join(json.loads(record["detailed_categories"]))
                        )

                
                with col2:
                    if st.button(
                        "Delete",
                        key=f"delete_{record['id']}"
                    ):
                        st.session_state["confirm_delete_id"] = int(record["id"])
                        st.rerun()

                if st.session_state.get("confirm_delete_id") == int(record["id"]):
                    st.warning(f"Are you sure you want to delete {record['filename']}?")

                    confirm_col1, confirm_col2 = st.columns(2)

                    with confirm_col1:
                        if st.button("Yes, Delete", key=f"confirm_{record['id']}"):
                            delete_document(record["id"])
                            st.session_state.pop("confirm_delete_id", None)
                            st.rerun()

                    with confirm_col2:
                        if st.button("Cancel", key=f"cancel_{record['id']}"):
                            st.session_state.pop("confirm_delete_id", None)
                            st.rerun()
                            
        st.divider()
        st.subheader("Deleted History")

        
        deleted_history = get_deleted_document_history()

        if deleted_history.empty:
            st.info("No deleted documents available.")

        else:
            if st.button("Clear All Deleted History"):
                st.session_state["confirm_clear_history"] = True
                st.rerun()

            if st.session_state.get("confirm_clear_history", False):
                st.warning("Are you sure you want to permanently delete all deleted records?")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Yes, Clear All"):
                        clear_deleted_history()
                        st.session_state["confirm_clear_history"] = False
                        st.success("Deleted history cleared successfully!")
                        st.rerun()

                with col2:
                    if st.button("Cancel"):
                        st.session_state["confirm_clear_history"] = False
                        st.rerun()

            st.write("Deleted documents:", len(deleted_history))

            
            for _, record in deleted_history.iterrows():
                with st.expander(
                    f"{record['filename']} — Deleted: {record['deleted_date']}"
                ):
                    st.write(
                        "**Main Categories:**",
                        ", ".join(json.loads(record["main_categories"]))
                    )

                    st.write(
                        "**Detailed Categories:**",
                        ", ".join(json.loads(record["detailed_categories"]))
                    )

                    st.write("**Original Upload Date:**", record["upload_date"])

                    if st.button(
                        "Restore Document",
                        key=f"restore_{record['id']}"
                    ):
                        restore_document(record["id"])
                        st.success("Document restored successfully!")
                        st.rerun()






# ---------------- DOCUMENT CHATBOT ----------------
elif page == "Document Chatbot":
    st.header("Chat with Your PDF Documents")
    st.write("Upload PDFs and ask questions about their contents. Answers are based only on PDFs processed here.")

    uploaded_pdfs = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="chatbot_pdf_upload"
    )

    if uploaded_pdfs:
        st.write("Selected documents:", len(uploaded_pdfs))
        for pdf in uploaded_pdfs:
            st.write("📄", pdf.name)

        if st.button("Process Documents", key="process_chat_pdfs"):
            with st.spinner("Extracting text and creating embeddings... This may take a few minutes."):
                try:
                    result = create_document_vectorstore(uploaded_pdfs)

                    if result is None:
                        st.error("No readable text was found in the selected PDFs.")
                    else:
                        vectorstore, llm = result
                        st.session_state["chat_vectorstore"] = vectorstore
                        st.session_state["chat_llm"] = llm
                        st.session_state["chat_messages"] = []
                        st.session_state["chat_pdf_names"] = [
                            pdf.name for pdf in uploaded_pdfs
                        ]
                        st.success("PDFs processed! You can now ask questions.")

                except Exception as e:
                    st.error(f"Document processing error: {e}")

    if "chat_vectorstore" in st.session_state:
        st.divider()
        st.subheader("Ask Questions")
        if st.button("Clear Chat"):
            st.session_state["chat_messages"] = []
            st.rerun()
        st.caption("This chat uses only the PDFs processed in this session.")
        st.write(
            "Currently processed:",
            ", ".join(st.session_state.get("chat_pdf_names", []))
        )

        # Display previous chat messages, including their source PDF pages.
        for message in st.session_state.get("chat_messages", []):
            with st.chat_message(message["role"]):
                st.write(message["content"])

               
        question = st.chat_input("Ask something about your uploaded PDFs...")

        if question:
            st.session_state["chat_messages"].append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching the documents and generating an answer..."):
                    try:
                        answer, sources = answer_document_question(
                            st.session_state["chat_vectorstore"],
                            st.session_state["chat_llm"],
                            question
                        )

                        st.write(answer)

                       

                        st.session_state["chat_messages"].append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })

                    except Exception as e:
                        st.error(f"Unable to answer the question: {e}")
                    