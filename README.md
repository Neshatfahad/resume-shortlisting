## Resume Shortlisting Automation
A Python-based system for automated resume parsing, semantic matching, and shortlisting against job descriptions. This tool leverages transformer models and LLMs for robust, explainable candidate screening.

📁 Project Structure
text
D:\RSA\
│
├── resume_folder/           # Store all candidate resumes (PDF)
│   ├── Structured_Resume_1_Arav.pdf
│   ├── Structured_Resume_2_Vivaan.pdf
│   ├── Structured_Resume_3_Aditya.pdf
│   ├── Structured_Resume_4_Krishna.pdf
│   ├── Structured_Resume_5_Ishaan.pdf
│   ├── Structured_Resume_6_Anaya.pdf
│   ├── Structured_Resume_7_Diya.pdf
│   ├── Structured_Resume_8_Myra.pdf
│   ├── Structured_Resume_9_Ananya.pdf
│   └── Structured_Resume_10_Kiara.pdf
│
├── resume/                  # Main code and modules
│   ├── resume_shortlister.py
│   ├── jd_parser_agent.py
│   ├── pyproject.toml
│   ├── .env
│   ├── .python-version
│   └── README.md
│
├── spacyenv/                # (env, optional)
├── __pycache__/             # (auto-generated)
└── resume_results_*.xlsx    # Output Excel files with results

📝 Sample Resumes Provided
The resume_folder directory contains a set of sample structured resumes in PDF format.
These are used for testing and demonstration purposes.
When you run the shortlisting script, it will automatically process all PDF files in this folder.

Included resumes:

Filename	Candidate Name
Structured_Resume_1_Arav.pdf	Arav
Structured_Resume_2_Vivaan.pdf	Vivaan
Structured_Resume_3_Aditya.pdf	Aditya
Structured_Resume_4_Krishna.pdf	Krishna
Structured_Resume_5_Ishaan.pdf	Ishaan
Structured_Resume_6_Anaya.pdf	Anaya
Structured_Resume_7_Diya.pdf	Diya
Structured_Resume_8_Myra.pdf	Myra
Structured_Resume_9_Ananya.pdf	Ananya
Structured_Resume_10_Kiara.pdf	Kiara
Location:
All resumes are located in the resume_folder directory at the root of the project.

Usage:

You can add, remove, or replace PDF files in resume_folder as needed.

The script will automatically process all PDF resumes present in this directory.

⚙️ Setup Instructions
Clone/Download the Repository

Place all code inside the resume directory.

Place all candidate resumes (PDFs) in resume_folder.

Install Dependencies

Ensure Python 3.8+ is installed.

Install required packages:

bash
pip install -r requirements.txt
requirements.txt contents:

text
transformers
torch
pandas
PyMuPDF
groq
python-dotenv

## Environment Variables

Create a .env file in the resume directory:

text
GROQ_API_KEY=your_groq_api_key_here
This key is required for LLM-based parsing and justification.

🚀 How It Works
1. Resume Parsing
All PDF resumes in resume_folder are read and text is extracted using PyMuPDF.

Each resume is parsed using a Groq LLM to extract structured fields:

Name, Email, Phone, Skills, Education, Experience, Certifications, Summary.

2. Job Description (JD) Processing
The job description (JD) is provided as input.

Core required skills are extracted from the JD using regex and heuristics.

3. Semantic Matching & Scoring
Each resume is scored on:

Relevant Experience: Years of experience, weighted by semantic similarity to the JD.

Skills Coverage: Semantic similarity of resume skills/experience to core JD skills.

Education Match: Semantic relevance of education section to JD.

Uses a cross-encoder transformer model (cross-encoder/ms-marco-MiniLM-L-12-v2) for semantic scoring.

4. LLM Justification
For each resume, a Groq LLM generates a justification/explanation for the match or rejection, strictly based on the JD requirements.

5. Output
Results are saved as an Excel file (e.g., resume_results_YYYYMMDD_HHMMSS.xlsx).

Each row contains: Resume name, scores, parsed fields, and justification.

💻 Usage
Place all resumes in resume_folder.

Run the main script:

bash
python resume_shortlister.py
You will be prompted (or can configure) to provide the JD text.

Check the output Excel file for ranked results and explanations.

🗂️ Key Files
File	Purpose
resume_shortlister.py	Main script: parsing, scoring, shortlisting, justification
jd_parser_agent.py	JD parsing and core skill extraction logic
.env	Stores API keys securely
pyproject.toml	Python project metadata
README.md	This documentation
⚡ Customizing the Flow
Skill Extraction: Edit extract_core_skills_from_jd() in jd_parser_agent.py for custom JD parsing.

Scoring Thresholds: Adjust interpret_score() and coverage thresholds for stricter/looser shortlisting.

Model Choice: Change MODEL_NAME in resume_shortlister.py for different transformer models.

🛠️ Troubleshooting
API Key Errors: Ensure .env is present and GROQ_API_KEY is valid.

PDF Extraction Issues: Ensure resumes are clear, text-based PDFs (not scanned images).

Performance: For large batches, consider batching or parallel processing.

🚀 Extending the Project
Add support for DOCX resumes.

Integrate with HR systems (ATS).

Enhance JD/resume parsing with more advanced LLMs or custom prompts.

📬 Contact
For questions or contributions, contact the maintainer or open an issue.