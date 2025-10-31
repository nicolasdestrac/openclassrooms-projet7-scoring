# tests/test_api.py
def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    js = r.json()
    assert js["status"] == "ok"
    assert js["n_input_columns"] >= 1

def test_schema(api_client):
    r = api_client.get("/schema")
    assert r.status_code == 200
    js = r.json()
    assert "input_columns" in js
    assert isinstance(js["input_columns"], list)

def test_predict_proba(api_client):
    payload = {"features": {"AMT_INCOME_TOTAL": 60000, "AMT_CREDIT": 120000}}
    r = api_client.post("/predict_proba", json=payload)
    assert r.status_code == 200
    js = r.json()
    assert 0.0 <= js["probability"] <= 1.0

def test_predict(api_client):
    payload = {"features": {"AMT_INCOME_TOTAL": 60000, "AMT_CREDIT": 120000}}
    r = api_client.post("/predict", json=payload)
    assert r.status_code == 200
    js = r.json()
    assert "probability" in js and "prediction" in js and "threshold" in js
    assert js["prediction"] in (0, 1)
