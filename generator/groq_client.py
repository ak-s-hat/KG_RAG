import os
import sys
from pathlib import Path

# Add root directory to sys.path to allow running this script directly
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from groq import Groq, RateLimitError
import time
from config import GENERATION_MODELS

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def execute_with_fallback(func, *args, **kwargs):
    """
    Execute a Groq API call with fallback to other models in GENERATION_MODELS
    if a RateLimitError occurs.
    """
    for model in GENERATION_MODELS:
        try:
            # Inject model into kwargs if the function expects it
            # But here we are calling client.chat.completions.create directly usually
            # So we need to handle how the API call is structured.
            # Actually, the simplest way is to pass the model to the API call.
            # Let's refactor to have the API call inside the loop.
            return func(model, *args, **kwargs)
        except RateLimitError as e:
            print(f"⚠️ Rate limit exceeded for {model}. Switching to next model...")
            time.sleep(1) # Brief pause
            continue
        except Exception as e:
            raise e
    
    raise RateLimitError("All models in fallback list exhausted.")

def extract_keywords(query: str, keywords: list):
    """
    Extract important keywords from the query using Groq.
    Returns a list of strings.
    """
    prompt = f"""
From the given list of keywords, extract all the keywords which are relevant to the given query.
Return ONLY the relevant keywords as a comma-separated list. No explanations.
All relevant keywords should be EXACTLY present in the list of keywords provided.
Note that these relevant keywords will be used to find information related to the query in a legal document.
#####
EXAMPLE 1:
- Query: 'Tell me about the benefits of Civil Union Partner and its benefits.'
- List of keywords: ['civil union', 'civil union partner', 'civil union partnership', 'drug addiction', 'due', 'dune buggy', 'duty']
- Relevant keywords: ['civil union partner', 'civil union', 'civil union partnership']
EXAMPLE 2:
- Query: 'How are medical predictive diagnostics improving?'
- List of keywords: ['healthcare', 'prediction', 'analysis', 'machine learning', 'drug abuse']
- Relevant keywords: ['healthcare', 'machine learning']
####
- Query: '{query}'
- List of keywords: {keywords}
- Relevant keywords:"""
    
    def _call_api(model):
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256
        )

    try:
        response = execute_with_fallback(_call_api)
        response_text = response.choices[0].message.content.strip()
        if not response_text:
            return []
        
        extracted = [kw.strip(" \"'[]") for kw in response_text.split(",")]
        return [kw for kw in extracted if kw]
    except Exception as e:
        print(f"❌ Error in extract_keywords: {e}")
        return []

def generate_answer(query: str, chunks: list):
    """
    Generate answer of query based on chunks.
    Returns a string
    """
    context = "\n".join([c['content'] for c in chunks])
    prompt = f"""Based on the context provided answer the query that follows.
The answer MUST be informative and simple enough for a common man to understand while being legally and technically correct.

If the given context does not contain answer to the query, respond with:
'Sorry, I could not find the answer to your question in the provided document.'
And also add a brief summary of the context provided
###
CONTEXT:
{context}
###

QUERY: {query}"""
    
    def _call_api(model):
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024
        )

    response = execute_with_fallback(_call_api)
    
    return response.choices[0].message.content
