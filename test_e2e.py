"""End-to-end test for L90 API."""
import requests
import json
import time

BASE = "http://localhost:8000"

# 1. Login 
r = requests.post(f"{BASE}/login", json={"username": "Batman", "password": "Joker"})
token = r.json()["token"]
print("1. Login: OK")

# 2. Upload doc
with open("test_upload.txt", "rb") as f:
    r = requests.post(
        f"{BASE}/upload",
        files={"file": ("test_upload.txt", f, "text/plain")},
        data={"collection": "user_private_collection", "owner": "Batman"},
        headers={"X-Token": token},
    )
data = r.json()
print(f"2. Upload: {r.status_code} - {data.get('chunks_stored', 'N/A')} chunks")

# 3. Query
print("3. Running query (may take 30-60s)...")
r = requests.post(
    f"{BASE}/query",
    json={
        "query": "What is the Schrodinger equation?",
        "mode": "STRICT",
        "user_id": "Batman",
        "workspace_id": "default_workspace",
    },
    headers={"X-Token": token},
    timeout=120,
)
print(f"   Status: {r.status_code}")
if r.status_code == 200:
    d = r.json()
    print(f"   Answer: {d['answer'][:300]}")
    print(f"   Confidence: {d['confidence_score']}")
    gr = d.get("grounding_report", {})
    print(f"   Verdict: {gr.get('final_verdict', 'N/A')}")
else:
    print(f"   Error: {r.text[:500]}")

# 4. Notebooks
r = requests.get(f"{BASE}/notebooks", headers={"X-Token": token})
print(f"4. Notebooks: {r.status_code}")

# 5. Workspaces
r = requests.get(f"{BASE}/workspaces", headers={"X-Token": token})
print(f"5. Workspaces: {r.status_code}")

print("\nAll tests done!")
