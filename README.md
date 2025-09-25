\# Lead Intent Scoring — Backend Engineer Assignment



A FastAPI backend that ingests an offer and a CSV of leads, runs a rule-based + AI scoring pipeline, and returns intent (High/Medium/Low) and a 0–100 score.



\## Files

\- `main.py` — FastAPI app with endpoints:

&nbsp; - `POST /offer` — save offer JSON

&nbsp; - `POST /leads/upload` — upload CSV (name,role,company,industry,location,linkedin\_bio)

&nbsp; - `POST /score` — run scoring pipeline

&nbsp; - `GET /results` — return JSON results

&nbsp; - `GET /results/csv` — download results CSV

\- `requirements.txt`

\- `sample\_leads.csv`

\- `tests/test\_rules.py` (optional)

\- `Dockerfile` (optional)



\## Setup (local)

Windows PowerShell:

```powershell

py -3 -m venv venv

.\\venv\\Scripts\\Activate

pip install -r requirements.txt

uvicorn main:app --reload --port 8000



