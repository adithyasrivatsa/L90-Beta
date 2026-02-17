"""Verification script for RAG Overhaul.

Tests:
1. STRICT mode query (should fail/warn without docs, or return strict answer)
2. PARTIAL mode query (should return friendly answer + checks)
3. Math verification trigger
"""

import requests
import json
import sys

API_URL = "http://localhost:8000"
TOKEN = "mock_token_for_testing" # The API uses a mock auth in the current state if not enforced, but let's see. 
# Actually app.py checks db.get_session_user. I need to login first.

def login():
    print("Logging in...")
    try:
        resp = requests.post(f"{API_URL}/login", json={"username": "user1", "password": "1234"})
        resp.raise_for_status()
        data = resp.json()
        print(f"Login success. Token: {data['token'][:10]}...")
        return data['token']
    except Exception as e:
        print(f"Login failed: {e}")
        return None

def test_query(token, query, mode):
    print(f"\n--- Testing mode: {mode} ---")
    print(f"Query: {query}")
    headers = {"x-token": token}
    payload = {
        "query": query,
        "mode": mode,
        "user_id": "user1",
        "workspace_id": "default_workspace"
    }
    
    try:
        resp = requests.post(f"{API_URL}/query", json=payload, headers=headers)
        if resp.status_code != 200:
            print(f"Query failed: {resp.status_code} - {resp.text}")
            return
            
        data = resp.json()
        print(f"Response Status: {resp.status_code}")
        print(f"Answer Start: {data['answer'][:100]}...")
        print(f"Confidence: {data['confidence_score']}")
        
        # Check new fields
        print(f"Code Verification: {bool(data.get('code_verification'))}")
        if data.get('code_verification'):
             print(f"  -> Success: {data['code_verification'].get('success')}")
             
        print(f"Deep Reasoning: {len(data.get('deep_reasoning', []))} items")
        if data.get('deep_reasoning'):
            print(f"  -> Insight: {data['deep_reasoning'][0].get('key_insight')}")
            
        print(f"LaTeX Equations: {len(data.get('latex_equations', []))} items")
        
    except Exception as e:
        print(f"Test failed: {e}")

def main():
    token = login()
    if not token:
        sys.exit(1)

    # 1. Test STRICT mode (should be strict, maybe insufficient info if no docs)
    test_query(token, "What is the speed of light?", "STRICT")

    # 2. Test PARTIAL mode with Math (should trigger math executor)
    # Asking a math question that requires calculation
    test_query(token, "Calculate the energy of an object with mass 5kg and velocity 10m/s using E=0.5mv^2", "PARTIAL")

if __name__ == "__main__":
    main()
