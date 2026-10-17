
import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# =====================================
# 2. SIDEBAR
# =====================================

with st.sidebar:
    st.title("📄 Resume Analyzer")
    st.write("AI-powered resume analysis")
    st.divider()

    st.write("### Features")
    st.write("✓ PDF text extraction")
    st.write("✓ Skill detection")
    st.write("✓ Education detection")
    st.write("✓ ML career prediction")
    st.write("✓ Downloadable report")

    st.divider()
    st.caption("Student project | Python + ML + Streamlit")

# =====================================
# 3. MAIN PAGE
# =====================================

st.title("📄 AI Resume Analyzer")
st.subheader("Understand your resume. Discover your skills.")

st.write(
    "Upload a PDF resume to identify listed skills, "
    "education keywords, and a predicted career category."
)

uploaded_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

# Skills checked by the program
skills = [
    "Python",
    "Java",
    "SQL",
    "Machine Learning",
    "C++",
    "JavaScript",
    "Pandas",
    "NumPy",
    "HTML",
    "CSS",
    "React",
    "Spring",
    "MySQL"
]

education_keywords = [
    "B.Tech",
    "Artificial Intelligence",
    "Data Science"
]

# =====================================
# 4. ANALYZE RESUME
# =====================================

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("🔍 Analyze Resume", type="primary"):

        try:
            # Read the uploaded PDF
            reader = PyPDF2.PdfReader(uploaded_file)

            resume_text = ""

            for page in reader.pages:
                resume_text += page.extract_text() or ""

            if not resume_text.strip():
                st.error(
                    "No readable text found. "
                    "Please upload a text-based PDF."
                )

            else:
                # -----------------------------
                # 5. DETECT SKILLS
                # -----------------------------

                found_skills = []

                for skill in skills:
                    if skill.lower() in resume_text.lower():
                        found_skills.append(skill)

                # -----------------------------
                # 6. CALCULATE SKILL MATCH SCORE
                # -----------------------------

                score = (
                    len(found_skills) / len(skills)
                ) * 100

                # -----------------------------
                # 7. DETECT EDUCATION
                # -----------------------------

                education = []

                for item in education_keywords:
                    if item.lower() in resume_text.lower():
                        education.append(item)

                # -----------------------------
                # 8. DISPLAY RESULTS
                # -----------------------------

                st.divider()
                st.header("📊 Analysis Results")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Skill Match Score",
                    f"{score:.0f}%"
                )

                col2.metric(
                    "Skills Detected",
                    len(found_skills)
                )

                col3.metric(
                    "Education Keywords",
                    len(education)
                )

                st.progress(int(score))

                left, right = st.columns(2)

                with left:
                    st.subheader("✅ Skills Found")

                    if found_skills:
                        for skill in found_skills:
                            st.write(f"• {skill}")
                    else:
                        st.write("No listed skills detected.")

                with right:
                    st.subheader("🎓 Education")

                    if education:
                        for item in education:
                            st.write(f"• {item}")
                    else:
                        st.write(
                            "No listed education keywords detected."
                        )

                # -----------------------------
                # 9. MACHINE LEARNING PREDICTION
                # -----------------------------

                st.subheader("🤖 Predicted Career Category")

                data = pd.read_csv("resume_data.csv")

                X = data["skills"].astype(str)
                y = data["category"]

                vectorizer = CountVectorizer()
                X_vectorized = vectorizer.fit_transform(X)

                model = MultinomialNB()
                model.fit(X_vectorized, y)

                resume_for_prediction = [
                    " ".join(found_skills)
                ]

                resume_vectorized = vectorizer.transform(
                    resume_for_prediction
                )

                prediction = model.predict(
                    resume_vectorized
                )[0]

                st.info(f"Predicted Category: {prediction}")

                # -----------------------------
                # 10. CREATE DOWNLOADABLE REPORT
                # -----------------------------

                report = f"""
AI RESUME ANALYZER REPORT
=========================

Resume File: {uploaded_file.name}

SKILL MATCH SCORE
-----------------
{score:.0f}%

SKILLS DETECTED
---------------
{", ".join(found_skills) if found_skills else "None"}

EDUCATION KEYWORDS
------------------
{", ".join(education) if education else "None"}

PREDICTED CAREER CATEGORY
-------------------------
{prediction}

IMPORTANT NOTE
--------------
This is an educational demonstration.
The score measures matches against a fixed list
of {len(skills)} skills.

The career prediction depends on a small training
dataset and should not be treated as professional
recruitment advice.
"""

                st.download_button(
                    label="📥 Download Analysis Report",
                    data=report,
                    file_name="resume_analysis_report.txt",
                    mime="text/plain"
                )

                st.caption(
                    "Educational demonstration only. "
                    "Results depend on the fixed skill list "
                    "and the small ML training dataset."
                )

        except FileNotFoundError:
            st.error(
                "resume_data.csv is missing. "
                "Keep it in the same folder as app.py."
            )

        except Exception as e:
            st.error(f"Analysis error: {e}")

else:
    st.info("👆 Upload a PDF resume to get started.")

# =====================================
# 11. FOOTER
# =====================================

st.divider()

st.caption(
    "AI Resume Analyzer | Built with Python, Pandas, "
    "scikit-learn, PyPDF2 and Streamlit"
)
