import os
import fitz  # PyMuPDF
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from groq import Groq
import re
from collections import Counter
from dotenv import load_dotenv # Import dotenv
 
# Ensure the jd_parser_agent.py file is in the same directory or accessible
from jd_parser_agent import JDParserAgent # Import the new agent
 
# Load environment variables
load_dotenv()
 
# === SETTINGS ===
RESUME_FOLDER = "../resume_folder"
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
# Use environment variable for API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please create a .env file or set the environment variable.")
GROQ_MODEL = "llama3-8b-8192"
 
# === LOAD MODELS ===
print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
groq_client = Groq(api_key=GROQ_API_KEY)
print("Models loaded.\n")
 
# === HELPER FUNCTIONS ===
def extract_text_from_pdf(filepath):
    try:
        pdf = fitz.open(filepath)
        full_text = ""
        page_texts = []
        for i, page in enumerate(pdf):
            text = page.get_text().strip()
            if text:
                full_text += text + "\n"
                page_texts.append({"id": f"{os.path.basename(filepath)}_page_{i+1}", "text": text})
        return full_text, page_texts
    except Exception as e:
        print(f"❌ Failed to open {filepath}: {e}")
        return "", []
 
def get_groq_parsed_resume(resume_text):
    prompt = (
        "You are a resume parsing assistant.\n"
        "Extract the following information as a JSON object:\n"
        "- Name\n- Email\n- Phone\n- Skills\n- Education\n- Experience\n- Certifications\n- Summary\n\n"
        f"Resume Text:\n{resume_text}\n\n"
        "Provide only a JSON output."
    )
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured resume information."},
                {"role": "user", "content": prompt}
            ]
        )
        import json, re
        content = response.choices[0].message.content
        # Extract JSON from markdown/code block if present
        match = re.search(r"```(?:json)?\n?(.*?)```", content, re.DOTALL)
        json_str = match.group(1) if match else content
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"parsing_error": "Failed to parse JSON from Groq response", "raw_response": content}
    except Exception as e:
        return {"error": f"Groq API error during parsing: {e}"}
 
def get_groq_justification(query, resume_text):
    # This function should use the original user query for justification
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict resume screening assistant. If a resume does not clearly meet the core technical requirements of the job description, say so directly. Do not attempt to justify mismatches."},
                {"role": "user", "content": f"Job Description: {query}\n\nResume:\n{resume_text}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Groq API error: {e}"
 
def score_text_pair(query, text):
    if not text or len(text.strip()) == 0:
        return 0.0
    inputs = tokenizer([(query, text)], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    score = logits.squeeze().item()
    return score
 
def interpret_score(score):
    # Adjusted thresholds based on user request (30% or above for higher match level)
    # Using raw score for interpretation, which corresponds to matching percent / 100
    if score >= 0.6:
        return "High Match"
    elif score >= 0.3:
        return "Medium Match"
    else:
        return "Low Match"
 
def extract_core_skills_from_jd(jd_text):
    """
    Extracts a set of core required skills from the job description.
    Prioritizes lines after 'Required Skills', 'Qualifications', etc.
    Uses the original query for precise extraction.
    """
    core_skills = set()
    sections = re.split(r'(Required Skills|Qualifications|Skills and Qualifications|Requirements|Key Skills):?', jd_text, flags=re.IGNORECASE)
    # Look for skills explicitly listed after headings
    if len(sections) > 1:
        # Take the text after the first matched heading
        text_after_heading = sections[2] if len(sections) > 2 else sections[1] # Fix index here
        for skill_line in re.split(r'[\n;-]', text_after_heading):
             # Split by commas, look for phrases
            candidates = [s.strip() for s in re.split(r',|\[|\(|\)|\]', skill_line) if s.strip()]
            core_skills.update([c.lower() for c in candidates])
 
    # Fallback/supplement: extract capitalized words (likely skills/technologies)
    capitalized = re.findall(r'\b([A-Z][a-zA-Z0-9\+\#\.]+)\b', jd_text)
    core_skills.update([word.lower() for word in capitalized if len(word) > 2 and word not in ['The', 'And', 'With', 'For', 'In', 'On']]) # Simple filter for common words
 
    # Consider only reasonably long/specific terms
    return {skill for skill in core_skills if len(skill) > 2}
 
def relevant_experience_years(parsed_experience, core_skills, expanded_query):
    """
    Returns the number of years of experience, weighted by how semantically relevant
    the experience entry is to the expanded query/JD domain.
    """
    if not parsed_experience:
        return 0.0
    total_relevant_years = 0.0
    experience_list = parsed_experience if isinstance(parsed_experience, list) else [parsed_experience]
 
    for entry in experience_list:
        text = ''
        years = 0.0
        if isinstance(entry, dict):
            # Prioritize 'years', 'duration' keys for years, take other values for text
            text = ' '.join([str(v) for k, v in entry.items() if k.lower() not in ['years', 'duration']])
            years_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?|yoe)', str(entry.get('years', '') or entry.get('duration', '')), re.IGNORECASE)
            if years_match:
                 years = float(years_match.group(1))
            # If years not found in specific keys, try text
            if years == 0.0:
                years_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?|yoe)', text, re.IGNORECASE)
                if years_match:
                    years = float(years_match.group(1))
 
        elif isinstance(entry, str):
            text = entry
            years_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?|yoe)', text, re.IGNORECASE)
            if years_match:
                years = float(years_match.group(1))
 
        if years > 0 and text:
            # Calculate semantic relevance of this experience entry to the expanded query
            if expanded_query:
                relevance_score = score_text_pair(expanded_query, text) # Use expanded query
            else:
                 relevance_score = 0.0
 
            # Simple weighting: relevance_score is usually between -1 and 1. Clamp and scale.
            # Multiply years by the clamped and scaled relevance score
            weighted_years = years * max(0.0, (relevance_score + 1) / 2.0) # Scale relevance to 0-1
            total_relevant_years += weighted_years
 
    # Normalize total relevant years by a reasonable cap (e.g., 10 years relevant = max score)
    # This normalization needs to happen AFTER summing all relevant years entries
    # Let's do a simpler normalization here: cap at a high value like 15 to avoid over-inflating
    return min(total_relevant_years, 15.0) # Cap total relevant years at 15 for score normalization later
 
def semantic_education_score(expanded_query, education_data):
    """
    Calculates a semantic score for the education section against the expanded query.
    """
    if not education_data or not expanded_query:
        return 0.0
 
    education_text = ''
    if isinstance(education_data, list):
        for entry in education_data:
            if isinstance(entry, dict):
                education_text += ' '.join([str(v) for v in entry.values()]) + '\n'
            elif isinstance(entry, str):
                education_text += entry + '\n'
    elif isinstance(education_data, str):
        education_text = education_data
 
    if education_text.strip():
        # Use expanded query for semantic relevance of education
        semantic_score = score_text_pair(expanded_query, education_text)
        # Normalize semantic relevance score (-1 to 1 range) to 0 to 1
        return max(0.0, (semantic_score + 1) / 2.0)
    else:
        return 0.0
 
def has_sufficient_core_skill_coverage(core_skills, parsed_skills, parsed_experience):
    """
    Checks if the resume has sufficient semantic coverage of the core required skills.
    Calculates semantic similarity of combined skills/experience text against each core skill.
    Returns True if average similarity is above a threshold.
    Note: This function still scores against *individual core skills*, not the expanded query.
    """
    if not core_skills:
        return True # No specific skills required, consider covered
 
    skill_exp_text = ''
    if isinstance(parsed_skills, list):
        skill_exp_text += ', '.join([str(s) for s in parsed_skills])
    elif isinstance(parsed_skills, str):
         skill_exp_text += parsed_skills
 
    if isinstance(parsed_experience, list):
        skill_exp_text += ' '.join([' '.join(str(v) for v in entry.values()) if isinstance(entry, dict) else str(entry) for entry in parsed_experience])
    elif isinstance(parsed_experience, str):
        skill_exp_text += parsed_experience
 
    if not skill_exp_text.strip():
        # If there's no text in skills/experience, coverage depends on if core_skills is empty
        return not core_skills # If no core skills needed, return True, else False
 
    # Calculate average semantic similarity of skill_exp_text to core skills
    similarity_scores = []
    for core_skill in list(core_skills):
         # Score the resume section against the individual core skill
         score = score_text_pair(core_skill, skill_exp_text)
         similarity_scores.append(score)
 
    # Check if average similarity is above a threshold (e.g., 0.2) AND if most core skills have some positive score
    if not similarity_scores:
        return not core_skills # If no core skills needed, return True, else False
 
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    # Check how many core skills have a similarity score > 0 (indicating some relation)
    positive_scores_count = sum(1 for score in similarity_scores if score > 0.1) # Slightly raise threshold for positive score
 
    # Require both a reasonable average score AND coverage of a significant portion of skills
    # Thresholds might need tuning (e.g., avg_similarity > 0.2, positive_scores_count >= len(core_skills) * 0.6)
    # Let's use: average similarity > 0.1 AND at least 50% of skills have a positive score
    return avg_similarity > 0.15 and positive_scores_count >= len(core_skills) * 0.5
 
def rerank_resumes(original_query, expanded_query):
    all_results = []
    best_resume = None
    best_score = float('-inf')
 
    print(f"\nAnalyzing resumes for query:\n👉 '{original_query}'\n")
    # print(f"🔍 Expanded Query Terms: {expanded_query}\n") # Optional: print expanded query
 
    # Use original query for core skill extraction
    core_skills = extract_core_skills_from_jd(original_query)
    print(f"[DEBUG] Core required skills extracted from JD: {sorted(list(core_skills))}\n") # Keep debug optional
 
    for filename in os.listdir(RESUME_FOLDER):
        if not filename.lower().endswith('.pdf'):
            continue
        filepath = os.path.join(RESUME_FOLDER, filename)
        full_text, page_texts = extract_text_from_pdf(filepath)
        if not page_texts:
            print(f"⚠️ No text found in {filename}")
            continue
 
        # Parse resume
        parsed = get_groq_parsed_resume(full_text)
        skills = parsed.get('Skills', [])
        experience = parsed.get('Experience', [])
        education = parsed.get('Education', []) # Get education data
 
        # --- New/Improved Scoring Components ---
 
        # 1. Relevant Experience Score (weighted by semantic relevance to expanded query)
        # Pass expanded_query to the function
        relevant_yoe = relevant_experience_years(experience, core_skills, expanded_query)
        # Normalize relevant years to a score (e.g., 5 years relevant = 0.5, 10+ relevant = 1.0)
        # Use a different normalization curve if desired. Current: relevant_yoe / 10.0 capped at 1
        exp_score = min(1.0, relevant_yoe / 10.0) # Normalize total relevant YOE by a max (e.g., 10 years)
 
        # 2. Semantic Skills Score (overall semantic relevance of skills to expanded query)
        skills_text_combined = ', '.join([str(s) for s in skills]) if isinstance(skills, list) else str(skills)
        # Use expanded query for semantic relevance of skills
        semantic_skills_relevance = score_text_pair(expanded_query, skills_text_combined)
        # Normalize semantic relevance score (-1 to 1 range) to 0 to 1
        skills_score = max(0.0, (semantic_skills_relevance + 1) / 2.0) # Use normalized semantic score for skills
 
        # 3. Semantic Education Score (relevance of education to expanded query)
        # Pass expanded_query to the function
        edu_score = semantic_education_score(expanded_query, education)
        # edu_score is already normalized 0-1 inside the function
 
        # 4. Top Page Semantic Match (relevance of best page to expanded query)
        # Use expanded query for page scoring
        pairs = [(expanded_query, doc['text']) for doc in page_texts]
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = model(**inputs).logits
        scores = logits.squeeze().tolist()
        if isinstance(scores, float):
            scores = [scores]
        for i, score in enumerate(scores):
            page_texts[i]['score'] = score
        top_page = max(page_texts, key=lambda x: x['score'])
        top_page_score = top_page['score']
        # Normalize top page score (-1 to 1 range) to 0 to 1 for consistent weighting
        top_page_score_norm = max(0.0, (top_page_score + 1) / 2.0)
 
        # --- Combined Weighted Final Score (Adjusted Weights: Skills > YOE > Education) ---
        # New Weights: Semantic Skills:Relevant YOE:Semantic Education:Top Page = 45:40:10:5
        final_score = (0.50 * skills_score) + (0.35 * exp_score) + (0.10 * edu_score) + (0.05 * top_page_score_norm)
 
        # --- Strict Penalization (Check if core skills are covered semantically) ---
        # Check for sufficient semantic coverage of core skills (uses individual core skills)
        has_sufficient_coverage = has_sufficient_core_skill_coverage(core_skills, skills, experience)
        penalized = False
        if not has_sufficient_coverage:
            # If key skills aren't sufficiently covered, heavily penalize or cap the score
            final_score = min(final_score, 0.3) # Cap at a low value if core skills not covered
            penalized = True
 
        all_results.append({
            'filename': filename,
            'score': final_score,
            'match_level': interpret_score(final_score),
            'parsed': parsed,
            'sem_skills_score': skills_score, # Store the component scores (normalized)
            'rel_exp_score': exp_score,
            'sem_edu_score': edu_score,
            'sem_top_page_score': top_page_score_norm,
            'relevant_years': relevant_yoe, # Store the raw relevant years before normalization
            'core_skills_coverage_sufficient': has_sufficient_coverage,
            'penalized': penalized,
            'top_page_text': top_page['text'],
            'core_skills_extracted': list(core_skills) # Store extracted core skills for debug/export
        })
 
        if final_score > best_score:
            best_score = final_score
            best_resume = all_results[-1]
 
    # Print all scores
    print("\n📊 Scores for all resumes:")
    for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
        final_score = result['score'] # Use the actual calculated score
        matching_percent = max(0, min(1, final_score)) * 100 # Clamp to 0-100 for display
        coverage_note = " [LOW SKILL COVERAGE]" if not result['core_skills_coverage_sufficient'] else ""
        match_decision = "ACCEPTED" if final_score >= 0.3 else "REJECTED" # Accept/Reject based on 0.3 threshold

        print(f" - {result['filename']}: {final_score:.2f} ({result['match_level']}) | Matching Percent: {matching_percent:.0f}% {match_decision}{coverage_note}")
 
    # Print best match
    if best_resume:
        best_final_score = best_resume['score']
        best_matching_percent = max(0, min(1, best_final_score)) * 100 # Clamp to 0-100 for display
        best_match_decision = "ACCEPTED" if best_final_score >= 0.3 else "REJECTED" # Accept/Reject based on 0.3 threshold

        # Use original_query for justification
        justification = get_groq_justification(original_query, best_resume['top_page_text'])
        print("\n✅ Best Matching Resume:")
        print(f"   📁 File     : {best_resume['filename']}")
        print(f"   ⭐ Score    : {best_final_score:.2f} ({best_matching_percent:.0f}%) [{best_match_decision}]") # Show decision in best match
        print(f"   🧠 Match    : {best_resume['match_level']}")
        coverage_note_best = " [LOW SKILL COVERAGE]" if not best_resume['core_skills_coverage_sufficient'] else ""
        if coverage_note_best:
             print(f"   ⚠️ Note: {coverage_note_best.strip()}")
        print(f"   📝 Top Page : {best_resume['top_page_text'][:200]}...")
        # Print relevant YOE and extracted core skills for best match (optional debug)
        # print(f"   Relevant YOE: {best_resume['relevant_years']:.2f}")
        # print(f"   Core Skills (Extracted from JD): {sorted(best_resume['core_skills_extracted'])}")
        print(f"   🧾 Justification by Groq:\n{'-'*40}\n{justification}\n{'-'*40}") # Justification uses original query
        # print(f"   📄 Parsed Data: {best_resume['parsed']}") # Keep parsed data print optional
    else:
        print("❌ No suitable resume found.")
 
    
# === ENTRY POINT ===
if __name__ == "__main__":
    # Initialize the JD Parser Agent
    try:
        jd_parser_agent = JDParserAgent()
    except ValueError as e:
        print(f"Error initializing JD Parser Agent: {e}")
        print("Please ensure you have a .env file with GROQ_API_KEY or the environment variable is set.")
        exit()
 
    while True:
        # Get user query
        user_query = input("\nEnter Job Description or Query (press Enter to exit): ").strip()
 
        # Exit the loop if the query is empty
        if not user_query:
            print("Exiting...")
            break
 
        print("🔍 Expanding query with related terms...")
        # Parse query with the agent to get expanded terms
        expanded_terms_list = jd_parser_agent.parse_query(user_query)
        expanded_query = " ".join(expanded_terms_list) # Join terms into a single string
 
        if not expanded_query.strip():
             print("⚠️ Failed to expand query or agent returned no terms. Using original query for scoring.")
             expanded_query = user_query # Fallback to original query if expansion fails
        else:
             print(f"✨ Query expanded to: {expanded_query[:100]}... (and more terms)") # Print snippet of expanded query
 
 
        # Run the resume reranking with both queries
        rerank_resumes(user_query, expanded_query)