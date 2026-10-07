from fastapi.testclient import TestClient
from app.core.config import settings


def test_root_endpoint(client: TestClient):
    """
    Test root endpoint returns valid status, doc pointers, and disclaimer.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "project" in data
    assert data["project"] == settings.PROJECT_NAME
    assert "disclaimer" in data
    assert "docs_url" in data
    assert data["docs_url"] == "/docs"


def test_health_check_endpoint(client: TestClient):
    """
    Test /api/v1/health returns healthy status and metadata.
    """
    response = client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["version"] == settings.VERSION
    assert data["environment"] == settings.ENVIRONMENT


def test_system_info_endpoint(client: TestClient):
    """
    Test /api/v1/info returns non-diagnostic disclaimer and project details.
    """
    response = client.get(f"{settings.API_V1_STR}/info")
    assert response.status_code == 200
    data = response.json()
    assert data["project_title"] == settings.PROJECT_NAME
    assert data["system_type"] == settings.SYSTEM_TYPE
    assert "disclaimer" in data
    assert "clinical diagnoses" in data["disclaimer"]
    assert len(data["modalities_supported"]) == 3
    assert "Personalized Longitudinal Z-Score Deviation" in data["baseline_paradigm"]


def test_cors_headers(client: TestClient):
    """
    Test CORS headers on preflight / options request.
    """
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET"
    }
    response = client.options(f"{settings.API_V1_STR}/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_settings_configuration():
    """
    Test Settings model loaded properly with expected defaults.
    """
    assert settings.API_V1_STR == "/api/v1"
    assert settings.VERSION == "1.0.0"
    assert "http://localhost:5173" in settings.BACKEND_CORS_ORIGINS
    assert len(settings.CLINICAL_DISCLAIMER) > 20
