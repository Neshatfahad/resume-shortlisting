from groq import Groq
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class JDParserAgent:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY in environment")
        self.client = Groq(api_key=api_key)
        
        # Corrected system prompt using triple quotes for multi-line string
        self.system_prompt = """You are an expert JD parser agent. Your task is to expand natural language queries 
into comprehensive lists of related terms, skills, and experience levels. Follow these rules:

1. Identify core concepts and technical terms
2. Include acronyms and full forms
3. Add related frameworks/libraries
4. Include experience variations (numerical and textual)
5. Add job title variations
6. Include adjacent technical skills
7. Output ONLY valid JSON array format.

Example Input: 'AI and Data worker with 5 years of experience in ML'
Example Output: ["Machine Learning", "ML", "Artificial Intelligence", "AI", "Data Science", "Python", "TensorFlow"]
"""

    def parse_query(self, query: str):
        try:
            completion = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.3
            )

            response = completion.choices[0].message.content

            # Try parsing the response assuming it's a valid JSON array string
            result = json.loads(response)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "terms" in result:
                return result["terms"]
            else:
                raise ValueError("Unexpected JSON structure")
        except Exception as e:
            print(f"Error processing query: {e}")
            return []

# === USAGE EXAMPLE ===
if __name__ == "__main__":
    agent = JDParserAgent()
    query = input("Enter your job description query: ") 
    results = agent.parse_query(query)

    print("🔍 Expanded terms:")
    for term in results:
        print(f"- {term}")
