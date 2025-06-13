import google.generativeai as genai

gemini_api_key = st.secrets["api_key"]
genai.configure(api_key= gemini_api_key)

def LLM(data):
  prompt = f"""
You are a car comparison assistant. Convert the following car recommendations {data} \n into STRICT comparison format:

The 4 points must be bullet points STRICTLY.

[Car A Name] vs [Car B Name]

[Car A Name]

1. [Insightful point about a key difference, such as safety, features, performance, etc.]

2. [Another relevant difference, could be design, reliability, economy, etc.]

3. [Another comparison that adds value, such as driving experience, comfort, etc.]

4. [Relevant point about overall value, suitability for a particular need, or market reputation]

[Car B Name]

1. [Insightful point about a key difference, such as safety, features, performance, etc.]

2. [Another relevant difference, could be design, reliability, economy, etc.]

3. [Another comparison that adds value, such as driving experience, comfort, etc.]

4. [Relevant point about overall value, suitability for a particular need, or market reputation]

Strict Rules:
Show it properly in bullet list. 

Ensure that the output is properly shown.

No intros, summaries, explanations, or extra text. Only the 4 bullet points per car.

Car B is always "Your Car" if not named.

Focus only on the most relevant differences based on the data provided.

Do not infer or assume missing data — use only the information provided.

If fewer than 4 meaningful differences exist, only include the strongest points. Avoid filler.

Each bullet point must be clear, short, and direct (maximum 5-6 words).

No general statements or fluff — every point must highlight a specific difference.

Prioritize key categories: safety, features, performance, design, reliability, economy, comfort, market reputation, driving experience, suitability.

Do not rephrase the data — extract and present only the clear comparisons.

Stay strictly within format. No deviations.

Do not include brackets in the bulletin points. 

Only reflect the data provided, no assumptions or additional context.


All the points 4 or anything less than that must be in bulletins. 
---"""

  response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
  return response.text