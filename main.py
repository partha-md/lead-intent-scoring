# main.py
"""
FastAPI backend for the Backend Engineer Hiring Assignment.

Endpoints:
- POST /offer           : Accepts JSON offer/product details (stored in memory)
- POST /leads/upload    : Accepts CSV upload (name,role,company,industry,location,linkedin_bio)
- POST /score           : Runs scoring pipeline (rule layer + AI layer) and stores results
- GET  /results         : Returns scored results as JSON
- GET  /results/csv     : Optional: download results CSV
Notes:
- If OPENAI_API_KEY is set in env, the app will attempt to call OpenAI's chat API.
- If no key (or API call fails), a local fallback AI classifier will be used.
"""

import os
import io
import csv
import json
import re
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Attempt to import openai. If not available or API key not set we'll fallback.
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    openai = None
    OPENAI_AVAILABLE = False

app = FastAPI(title="Lead Intent Scoring - Assignment")

# In-memory storage (simple for the assignment)
OFFER: Optional[Dict[str, Any]] = None
LEADS: List[Dict[str, str]] = []
RESULTS: List[Dict[str, Any]] = []

# ========== Pydantic model for /offer ==========
class Offer(BaseModel):
    name: str
    value_props: List[str] = []
    ideal_use_cases: List[str] = []


# ========== Helper functions for rule scoring ==========
def compute_rule_score(lead: Dict[str, str], offer: Dict[str, Any]) -> Tuple[int, str]:
    """
    Compute rule score (max 50) and return (score, short_reasoning)
    RULES:
      - Role relevance: decision maker (+20), influencer (+10), else 0
      - Industry match: exact ICP (+20), adjacent (+10), else 0
      - Data completeness: all fields present (+10)
    """
    role = (lead.get("role") or "").strip().lower()
    industry = (lead.get("industry") or "").strip().lower()

    # Role matching heuristics
    decision_keywords = [
        "ceo", "founder", "co-founder", "cofounder", "chief", "cto", "cfo",
        "vp", "vice president", "head of", "director", "owner", "president", "managing director"
    ]
    influencer_keywords = [
        "manager", "lead", "principal", "senior", "sr.", "sr", "associate",
        "executive", "analyst", "specialist", "coordinator"
    ]

    role_score = 0
    if any(k in role for k in decision_keywords):
        role_score = 20
        role_reason = "role = decision maker"
    elif any(k in role for k in influencer_keywords):
        role_score = 10
        role_reason = "role = influencer"
    else:
        role_score = 0
        role_reason = "role = other"

    # Industry matching heuristics
    industry_score = 0
    industry_reason = "industry no match"
    if offer and offer.get("ideal_use_cases"):
        lead_tokens = set(re.findall(r"\w+", industry))
        matched_exact = False
        matched_adjacent = False
        for ic in offer["ideal_use_cases"]:
            ic_l = (ic or "").strip().lower()
            # Exact-ish match rules
            if ic_l == industry or ic_l in industry or industry in ic_l:
                matched_exact = True
                break
            # Token intersection for adjacent
            ic_tokens = set(re.findall(r"\w+", ic_l))
            if lead_tokens & ic_tokens:
                matched_adjacent = True
        if matched_exact:
            industry_score = 20
            industry_reason = "industry exact match to ICP"
        elif matched_adjacent:
            industry_score = 10
            industry_reason = "industry adjacent to ICP"
        else:
            industry_score = 0
            industry_reason = "industry no match"

    # Data completeness
    completeness_score = 0
    required_fields = ["name", "role", "company", "industry", "location", "linkedin_bio"]
    if all((lead.get(f) and str(lead.get(f)).strip()) for f in required_fields):
        completeness_score = 10
        completeness_reason = "all fields present"
    else:
        completeness_score = 0
        completeness_reason = "missing fields"

    total = role_score + industry_score + completeness_score
    reason = f"{role_reason}; {industry_reason}; {completeness_reason}"

    # Cap to 50 (shouldn't exceed but keep safe)
    total = min(total, 50)
    return total, reason


# ========== AI layer: call OpenAI or fallback ==========
def ai_classify(lead: Dict[str, str], offer: Dict[str, Any], rule_score: int) -> Tuple[str, str, int]:
    """
    Returns (intent_label, ai_explanation, ai_points)
    - attempts to call OpenAI if available and OPENAI_API_KEY present
    - otherwise uses a deterministic fallback
    Mapping: High = 50, Medium = 30, Low = 10
    """
    # Build prompt content
    prompt_text = (
        "You are a helpful assistant that classifies a prospect's buying intent "
        "for a given product/offer. Respond with a single JSON object ONLY, e.g.:\n"
        '{"intent":"High","explain":"Fits ICP and decision maker."}\n\n'
        "Product/Offer:\n" + json.dumps(offer, ensure_ascii=False) + "\n\n"
        "Prospect:\n" + json.dumps(lead, ensure_ascii=False) + "\n\n"
        "Question: Classify intent (High/Medium/Low) and explain in 1-2 sentences."
    )

    # If OpenAI available and API key present, try calling; else fallback
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAIKEY")
    if OPENAI_AVAILABLE and api_key:
        try:
            openai.api_key = api_key
            # use chat completion
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You classify buyer intent."},
                    {"role": "user", "content": prompt_text},
                ],
                max_tokens=150,
                temperature=0.0
            )
            content = resp["choices"][0]["message"]["content"]
            # Try to extract JSON object from content
            json_text = _extract_json_like(content)
            parsed = json.loads(json_text)
            intent = parsed.get("intent", "").strip().capitalize()
            explain = parsed.get("explain", "").strip()
            if intent not in ("High", "Medium", "Low"):
                # fallback to local if model returned something odd
                raise ValueError("unexpected intent label from model")
        except Exception as e:
            # If anything goes wrong, fall back
            intent, explain = _fallback_ai_intent(lead, offer, rule_score)
    else:
        intent, explain = _fallback_ai_intent(lead, offer, rule_score)

    ai_points = {"High": 50, "Medium": 30, "Low": 10}.get(intent, 10)
    return intent, explain, ai_points


def _extract_json_like(text: str) -> str:
    """
    Extract the first {...} JSON substring from text.
    If not found, try to return the whole text (hoping it's JSON).
    """
    match = re.search(r"(\{(?:.|\n)*\})", text)
    if match:
        return match.group(1)
    return text


def _fallback_ai_intent(lead: Dict[str, str], offer: Dict[str, Any], rule_score: int) -> Tuple[str, str]:
    """
    Deterministic fallback that maps rule_score -> intent and crafts a short explanation.
    - rule_score >= 40 -> High
    - 20 <= rule_score < 40 -> Medium
    - else -> Low
    """
    if rule_score >= 40:
        intent = "High"
    elif rule_score >= 20:
        intent = "Medium"
    else:
        intent = "Low"

    # simple explanation using role and industry
    role = lead.get("role", "Unknown role")
    industry = lead.get("industry", "Unknown industry")
    explain = f"{role} in {industry} suggests {intent} intent based on rule signals."
    return intent, explain


# ========== API Endpoints ==========
@app.post("/offer")
async def post_offer(offer: Offer):
    """
    Save offer to memory. Overwrites previous offer.
    """
    global OFFER
    OFFER = offer.dict()
    return {"status": "ok", "offer": OFFER}


@app.post("/leads/upload")
async def upload_leads(file: UploadFile = File(...)):
    """
    Upload CSV file: columns: name,role,company,industry,location,linkedin_bio
    """
    global LEADS
    content = await file.read()
    try:
        s = content.decode("utf-8-sig")
    except Exception:
        s = content.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(s))
    required = {"name", "role", "company", "industry", "location", "linkedin_bio"}
    # normalize fieldnames
    fieldnames = {fn.strip().lower() for fn in reader.fieldnames or []}
    if not required.issubset(fieldnames):
        raise HTTPException(status_code=400, detail=f"CSV must contain columns: {sorted(required)}")
    # reset LEADS (assignment expects you to upload leads and then score that dataset)
    LEADS = []
    for row in reader:
        # standardize keys to lower-case names
        normalized = {k.strip().lower(): (v or "").strip() for k, v in row.items()}
        # Keep only required fields
        lead = {k: normalized.get(k, "") for k in required}
        LEADS.append(lead)
    return {"status": "ok", "count": len(LEADS)}


@app.post("/score")
async def run_scoring():
    """
    Run scoring pipeline on uploaded leads using current OFFER.
    Stores results in RESULTS and returns them.
    """
    global OFFER, LEADS, RESULTS
    if not OFFER:
        raise HTTPException(status_code=400, detail="No offer set. POST /offer first.")
    if not LEADS:
        raise HTTPException(status_code=400, detail="No leads uploaded. POST /leads/upload first.")

    RESULTS = []
    for lead in LEADS:
        rule_score, rule_reason = compute_rule_score(lead, OFFER)
        intent_label, ai_explain, ai_points = ai_classify(lead, OFFER, rule_score)
        final_score = rule_score + ai_points
        # cap final score 0-100
        final_score = max(0, min(100, final_score))
        result = {
            "name": lead.get("name"),
            "role": lead.get("role"),
            "company": lead.get("company"),
            "industry": lead.get("industry"),
            "location": lead.get("location"),
            "linkedin_bio": lead.get("linkedin_bio"),
            "intent": intent_label,
            "score": final_score,
            "reasoning": f"Rule: {rule_reason}. AI: {ai_explain}"
        }
        RESULTS.append(result)
    return {"status": "ok", "count": len(RESULTS), "results": RESULTS}


@app.get("/results")
async def get_results():
    return JSONResponse(content=RESULTS)


@app.get("/results/csv")
async def get_results_csv():
    """
    Export results as CSV. If no results, return 204.
    """
    if not RESULTS:
        raise HTTPException(status_code=400, detail="No results available. Run POST /score first.")
    headers = ["name", "role", "company", "industry", "location", "linkedin_bio", "intent", "score", "reasoning"]
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(headers)
    for r in RESULTS:
        writer.writerow([r.get(h, "") for h in headers])
    stream.seek(0)
    return StreamingResponse(iter([stream.getvalue()]),
                             media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=results.csv"})


# ========== Run the app for local dev ==========
if __name__ == "__main__":
    import uvicorn
    print("Starting app on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
