# """
# test_api.py - Tests for REST API endpoints
# =============================================
# """

# import pytest
# from fastapi.testclient import TestClient


# @pytest.fixture
# def client():
#     """Create a test client."""
#     from src.api.app import app
#     return TestClient(app)


# class TestHealthEndpoint:
#     """Tests for health check."""

#     def test_health_check(self, client):
#         response = client.get("/health")
#         assert response.status_code == 200
#         data = response.json()
#         assert "status" in data
#         assert data["status"] == "healthy"


# class TestPredictEndpoint:
#     """Tests for prediction endpoint."""

#     def test_predict_requires_text(self, client):
#         response = client.post("/api/v1/predict", json={})
#         assert response.status_code == 422

#     def test_predict_empty_text(self, client):
#         response = client.post("/api/v1/predict", json={"text": ""})
#         assert response.status_code == 422


# class TestBatchEndpoint:
#     """Tests for batch prediction endpoint."""

#     def test_batch_requires_texts(self, client):
#         response = client.post("/api/v1/batch", json={})
#         assert response.status_code == 422
