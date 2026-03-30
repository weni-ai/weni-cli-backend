"""
Agent evaluation endpoints for running evaluations via streaming.
"""

import logging
import os
import shutil
import tempfile
import threading
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

from app.api.v1.models.requests import RunEvaluationRequestModel
from app.core.response import CLIResponse, send_response

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_CONCURRENT_PER_PROJECT = 1
MAX_CONCURRENT_GLOBAL = 10


class _EvaluationRateLimiter:
    """In-memory concurrency limiter for evaluation runs."""

    def __init__(self) -> None:
        self._active: dict[str, int] = {}
        self._total_active: int = 0
        self._lock = threading.Lock()

    def acquire(self, project_uuid: str) -> bool:
        with self._lock:
            if self._total_active >= MAX_CONCURRENT_GLOBAL:
                return False
            if self._active.get(project_uuid, 0) >= MAX_CONCURRENT_PER_PROJECT:
                return False
            self._active[project_uuid] = self._active.get(project_uuid, 0) + 1
            self._total_active += 1
            return True

    def release(self, project_uuid: str) -> None:
        with self._lock:
            current = self._active.get(project_uuid, 0)
            if current <= 1:
                self._active.pop(project_uuid, None)
            else:
                self._active[project_uuid] = current - 1
            self._total_active = max(0, self._total_active - 1)


_rate_limiter = _EvaluationRateLimiter()


@router.post("")
async def run_evaluation(
    data: RunEvaluationRequestModel,
    authorization: Annotated[str, Header()],
    x_project_uuid: Annotated[str, Header()],
) -> StreamingResponse:
    if not _rate_limiter.acquire(x_project_uuid):
        raise HTTPException(
            status_code=429,
            detail="An evaluation is already running for this project. Please wait for it to finish.",
        )

    request_id = str(uuid4())
    logger.info(f"Processing evaluation run - request_id: {request_id}")

    async def response_stream() -> AsyncIterator[bytes]:
        work_dir = tempfile.mkdtemp(prefix="eval_")
        try:
            bearer_token = (
                authorization.replace("Bearer ", "")
                if authorization.startswith("Bearer ")
                else authorization
            )

            target_config = data.target.copy()
            target_config["weni_bearer_token"] = bearer_token
            target_config["weni_project_uuid"] = x_project_uuid

            if "LOG_LEVEL" in os.environ:
                os.environ["LOG_LEVEL"] = os.environ["LOG_LEVEL"].upper()

            from agenteval.evaluators import EvaluatorFactory
            from agenteval.targets import TargetFactory
            from agenteval.test import TestSuite

            evaluator_factory = EvaluatorFactory(config=data.evaluator)
            target_factory = TargetFactory(config=target_config)
            test_suite = TestSuite.load(data.tests, data.filter)

            num_tests = test_suite.num_tests
            test_names = [test.name for test in test_suite]

            started: CLIResponse = {
                "message": f"Starting evaluation with {num_tests} test(s)",
                "data": {
                    "num_tests": num_tests,
                    "test_names": test_names,
                },
                "success": True,
                "code": "EVALUATION_STARTED",
            }
            yield send_response(started, request_id=request_id)

            pass_count = 0
            all_tests = list(test_suite.tests)
            all_results = []

            for i, test in enumerate(test_suite):
                test_started: CLIResponse = {
                    "message": f"Running test: {test.name}",
                    "data": {
                        "test_name": test.name,
                        "test_index": i + 1,
                        "num_tests": num_tests,
                    },
                    "success": True,
                    "code": "EVALUATION_TEST_STARTED",
                }
                yield send_response(test_started, request_id=request_id)

                target = target_factory.create()
                evaluator = evaluator_factory.create(
                    test=test,
                    target=target,
                    work_dir=work_dir,
                )
                result = evaluator.run()

                if result.passed:
                    pass_count += 1

                all_results.append(result)

                conversation_data = []
                if result.conversation:
                    for role, message in result.conversation.messages:
                        conversation_data.append({
                            "role": role,
                            "message": message,
                        })

                test_completed: CLIResponse = {
                    "message": f"Test completed: {test.name}",
                    "data": {
                        "test_name": test.name,
                        "passed": result.passed,
                        "result": result.result,
                        "reasoning": result.reasoning,
                        "conversation": conversation_data,
                    },
                    "success": True,
                    "code": "EVALUATION_TEST_COMPLETED",
                }
                yield send_response(test_completed, request_id=request_id)

            from agenteval.summary import create_markdown_summary

            create_markdown_summary(
                work_dir,
                pass_count,
                num_tests,
                all_tests,
                all_results,
            )

            summary_content = ""
            summary_path = os.path.join(work_dir, "agenteval_summary.md")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    summary_content = f.read()

            completed: CLIResponse = {
                "message": "Evaluation completed",
                "data": {
                    "pass_count": pass_count,
                    "num_tests": num_tests,
                    "summary_content": summary_content,
                },
                "success": True,
                "code": "EVALUATION_COMPLETED",
            }
            yield send_response(completed, request_id=request_id)

        except Exception as e:
            logger.error(f"Error during evaluation: {e!s} - request_id: {request_id}")
            error_data: CLIResponse = {
                "message": "Error during evaluation",
                "data": {
                    "error": str(e),
                },
                "success": False,
                "code": "EVALUATION_ERROR",
            }
            yield send_response(error_data, request_id=request_id)
        finally:
            _rate_limiter.release(x_project_uuid)
            shutil.rmtree(work_dir, ignore_errors=True)

    return StreamingResponse(response_stream(), media_type="application/x-ndjson")
