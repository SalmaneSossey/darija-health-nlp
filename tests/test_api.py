from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_high_urgency() -> None:
    response = client.post("/predict", json={"message": "3ndi wje3 f sedri w di9 f nefs"})
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_specialty"] == "Cardiology"
    assert data["urgency"] == "high"
    assert "chest_pain" in data["symptoms"]
