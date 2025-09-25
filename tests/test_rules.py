from main import compute_rule_score

def test_compute_rule_score_decision_exact_complete():
    offer = {"ideal_use_cases": ["B2B SaaS mid-market"]}
    lead = {
        "name": "Test",
        "role": "CEO",
        "company": "X",
        "industry": "B2B SaaS mid-market",
        "location": "Delhi",
        "linkedin_bio": "Leader"
    }
    score, reason = compute_rule_score(lead, offer)
    assert score == 50
