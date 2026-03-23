"""Tests for agent evaluation endpoint."""

import json
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.core.config import settings
from app.main import app

TEST_TOKEN = "Bearer test-token"
TEST_PROJECT_UUID = str(uuid4())


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def api_path() -> str:
    return f"{settings.API_PREFIX}/v1/evaluations"


@pytest.fixture(scope="module")
def auth_headers() -> dict[str, str]:
    return {
        "Authorization": TEST_TOKEN,
        "X-Project-Uuid": TEST_PROJECT_UUID,
        "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
    }


@pytest.fixture
def evaluation_payload() -> dict[str, Any]:
    return {
        "evaluator": {
            "model": "claude-haiku-4_5-global",
            "aws_region": "us-east-1",
        },
        "target": {
            "type": "weni",
        },
        "tests": {
            "greeting": {
                "steps": ["Send a greeting message"],
                "expected_results": ["Agent responds with a greeting"],
            }
        },
    }


def parse_streaming_response(response: Any) -> list[dict[str, Any]]:
    result = []
    for line in response.iter_lines():
        if line:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result


def _make_mock_test(name: str = "greeting") -> MagicMock:
    mock_test = MagicMock()
    mock_test.name = name
    return mock_test


def _make_mock_result(passed: bool = True) -> MagicMock:
    mock_result = MagicMock()
    mock_result.passed = passed
    mock_result.result = "Test passed" if passed else "Test failed"
    mock_result.reasoning = "The agent responded correctly."
    mock_result.conversation = None
    return mock_result


def _make_mock_test_suite(mock_test: MagicMock) -> MagicMock:
    """Create a mock TestSuite that returns a fresh iterator on every __iter__ call."""
    mock_test_suite = MagicMock()
    mock_test_suite.num_tests = 1
    mock_test_suite.tests = [mock_test]
    mock_test_suite.__iter__ = MagicMock(side_effect=lambda: iter([mock_test]))
    return mock_test_suite


class TestEvaluationEndpoint:

    @pytest.fixture
    def mock_agenteval_success(self, mocker: MockerFixture) -> dict[str, MagicMock]:
        mock_test = _make_mock_test()
        mock_result = _make_mock_result(passed=True)
        mock_test_suite = _make_mock_test_suite(mock_test)

        mock_evaluator = MagicMock()
        mock_evaluator.run.return_value = mock_result

        mock_evaluator_factory = MagicMock()
        mock_evaluator_factory.create.return_value = mock_evaluator

        mock_target = MagicMock()
        mock_target_factory = MagicMock()
        mock_target_factory.create.return_value = mock_target

        mocker.patch(
            "app.api.v1.routers.evaluations.EvaluatorFactory",
            return_value=mock_evaluator_factory,
        )
        mocker.patch(
            "app.api.v1.routers.evaluations.TargetFactory",
            return_value=mock_target_factory,
        )
        mocker.patch(
            "app.api.v1.routers.evaluations.TestSuite",
            **{"load.return_value": mock_test_suite},
        )
        mocker.patch(
            "app.api.v1.routers.evaluations.create_markdown_summary",
        )

        return {
            "evaluator_factory": mock_evaluator_factory,
            "target_factory": mock_target_factory,
            "test_suite": mock_test_suite,
            "evaluator": mock_evaluator,
            "result": mock_result,
        }

    def test_evaluation_success(
        self,
        client: TestClient,
        api_path: str,
        auth_headers: dict[str, str],
        evaluation_payload: dict[str, Any],
        mock_agenteval_success: dict[str, MagicMock],
        mock_auth_middleware: None,
    ) -> None:
        response = client.post(api_path, json=evaluation_payload, headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK

        events = parse_streaming_response(response)

        codes = [e["code"] for e in events]
        assert "EVALUATION_STARTED" in codes
        assert "EVALUATION_TEST_STARTED" in codes
        assert "EVALUATION_TEST_COMPLETED" in codes
        assert "EVALUATION_COMPLETED" in codes

        started = next(e for e in events if e["code"] == "EVALUATION_STARTED")
        assert started["data"]["num_tests"] == 1
        assert started["success"] is True

        completed_test = next(e for e in events if e["code"] == "EVALUATION_TEST_COMPLETED")
        assert completed_test["data"]["test_name"] == "greeting"
        assert completed_test["data"]["passed"] is True

        completed = next(e for e in events if e["code"] == "EVALUATION_COMPLETED")
        assert completed["data"]["pass_count"] == 1
        assert completed["data"]["num_tests"] == 1

    def test_evaluation_injects_weni_credentials(
        self,
        client: TestClient,
        api_path: str,
        auth_headers: dict[str, str],
        evaluation_payload: dict[str, Any],
        mock_agenteval_success: dict[str, MagicMock],
        mock_auth_middleware: None,
    ) -> None:
        response = client.post(api_path, json=evaluation_payload, headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        mock_agenteval_success["target_factory"].create.assert_called_once()

    def test_evaluation_test_failure(
        self,
        client: TestClient,
        api_path: str,
        auth_headers: dict[str, str],
        evaluation_payload: dict[str, Any],
        mocker: MockerFixture,
        mock_auth_middleware: None,
    ) -> None:
        mock_test = _make_mock_test()
        mock_result = _make_mock_result(passed=False)
        mock_test_suite = _make_mock_test_suite(mock_test)

        mock_evaluator = MagicMock()
        mock_evaluator.run.return_value = mock_result

        mock_evaluator_factory = MagicMock()
        mock_evaluator_factory.create.return_value = mock_evaluator

        mock_target_factory = MagicMock()

        mocker.patch("app.api.v1.routers.evaluations.EvaluatorFactory", return_value=mock_evaluator_factory)
        mocker.patch("app.api.v1.routers.evaluations.TargetFactory", return_value=mock_target_factory)
        mocker.patch("app.api.v1.routers.evaluations.TestSuite", **{"load.return_value": mock_test_suite})
        mocker.patch("app.api.v1.routers.evaluations.create_markdown_summary")

        response = client.post(api_path, json=evaluation_payload, headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK

        events = parse_streaming_response(response)
        completed_test = next(e for e in events if e["code"] == "EVALUATION_TEST_COMPLETED")
        assert completed_test["data"]["passed"] is False

        completed = next(e for e in events if e["code"] == "EVALUATION_COMPLETED")
        assert completed["data"]["pass_count"] == 0

    def test_evaluation_error_handling(
        self,
        client: TestClient,
        api_path: str,
        auth_headers: dict[str, str],
        evaluation_payload: dict[str, Any],
        mocker: MockerFixture,
        mock_auth_middleware: None,
    ) -> None:
        mocker.patch(
            "app.api.v1.routers.evaluations.EvaluatorFactory",
            side_effect=ValueError("Invalid evaluator config"),
        )

        response = client.post(api_path, json=evaluation_payload, headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK

        events = parse_streaming_response(response)
        error_events = [e for e in events if e.get("success") is False]
        assert len(error_events) > 0
        assert "Invalid evaluator config" in str(error_events[-1])

    def test_evaluation_missing_auth(
        self,
        client: TestClient,
        api_path: str,
        evaluation_payload: dict[str, Any],
        mock_auth_middleware: None,
    ) -> None:
        headers = {"X-CLI-Version": settings.CLI_MINIMUM_VERSION}
        response = client.post(api_path, json=evaluation_payload, headers=headers)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_evaluation_with_filter(
        self,
        client: TestClient,
        api_path: str,
        auth_headers: dict[str, str],
        mocker: MockerFixture,
        mock_agenteval_success: dict[str, MagicMock],
        mock_auth_middleware: None,
    ) -> None:
        payload = {
            "evaluator": {"model": "claude-haiku-4_5-global", "aws_region": "us-east-1"},
            "target": {"type": "weni"},
            "tests": {
                "greeting": {
                    "steps": ["Send a greeting message"],
                    "expected_results": ["Agent responds with a greeting"],
                }
            },
            "filter": "greeting",
        }

        response = client.post(api_path, json=payload, headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK

        events = parse_streaming_response(response)
        codes = [e["code"] for e in events]
        assert "EVALUATION_STARTED" in codes

    def test_evaluation_rate_limit_per_project(
        self,
        client: TestClient,
        api_path: str,
        auth_headers: dict[str, str],
        evaluation_payload: dict[str, Any],
        mocker: MockerFixture,
        mock_auth_middleware: None,
    ) -> None:
        from app.api.v1.routers.evaluations import _rate_limiter

        project_uuid = auth_headers["X-Project-Uuid"]

        _rate_limiter.acquire(project_uuid)
        try:
            response = client.post(api_path, json=evaluation_payload, headers=auth_headers)
            assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        finally:
            _rate_limiter.release(project_uuid)
