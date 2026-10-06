"""Unit tests for the active agent run strategy helpers."""

import json
import logging

import pytest
from pytest_mock import MockerFixture

from app.services.runs import active_strategy
from app.services.runs.token_issuer import RunTokenIssuer


def _active_issuer() -> RunTokenIssuer:
    return RunTokenIssuer(
        authorized_project_uuid="11111111-1111-1111-1111-111111111111",
        user_email="dev@example.com",
        agent_key="agent_a",
        tool_key=None,
        run_type="active",
        request_id="request-id",
    )


@pytest.fixture
def definition() -> dict:
    return {
        "agents": {
            "agent_a": {
                "rules": {
                    "rule_x": {
                        "source": {"entrypoint": "main.RuleX"},
                        "template": "tpl_x",
                    }
                },
                "pre_processing": {
                    "source": {"entrypoint": "processing.PreProcessor"},
                },
            }
        }
    }


class TestBuildAgentResource:
    def test_builds_preprocessor_and_rules(self, definition: dict) -> None:
        resources = [
            ("agent_a:preprocessor_folder", b"pre"),
            ("agent_a:rule_x", b"rule"),
        ]
        model = active_strategy.build_agent_resource("agent_a", definition, resources)

        assert model.preprocessor is not None
        assert model.preprocessor.module_name == "processing"
        assert model.preprocessor.class_name == "PreProcessor"
        assert model.preprocessor.content == b"pre"

        assert len(model.rules) == 1
        rule = model.rules[0]
        assert rule.key == "rule_x"
        assert rule.module_name == "main"
        assert rule.class_name == "RuleX"
        assert rule.template == "tpl_x"
        assert rule.content == b"rule"

    def test_attaches_preprocessor_example(self, definition: dict) -> None:
        resources = [
            ("agent_a:preprocessor_folder", b"pre"),
            ("agent_a:rule_x", b"rule"),
            ("agent_a:preprocessor_example", b"{}"),
        ]
        model = active_strategy.build_agent_resource("agent_a", definition, resources)

        assert model.preprocessor_example == b"{}"

    def test_skips_other_agents(self, definition: dict) -> None:
        resources = [
            ("agent_a:preprocessor_folder", b"pre"),
            ("agent_a:rule_x", b"rule"),
            ("agent_b:rule_z", b"foreign"),
        ]
        model = active_strategy.build_agent_resource("agent_a", definition, resources)

        assert len(model.rules) == 1
        assert model.rules[0].key == "rule_x"

    def test_raises_when_agent_missing(self, definition: dict) -> None:
        with pytest.raises(ValueError, match="Could not find agent"):
            active_strategy.build_agent_resource("missing", definition, [])

    def test_raises_when_rule_missing_in_definition(self, definition: dict) -> None:
        resources = [
            ("agent_a:preprocessor_folder", b"pre"),
            ("agent_a:unknown_rule", b"rule"),
        ]
        with pytest.raises(ValueError, match="Could not find rule"):
            active_strategy.build_agent_resource("agent_a", definition, resources)

    def test_raises_when_preprocessor_missing(self, definition: dict) -> None:
        resources = [("agent_a:rule_x", b"rule")]
        with pytest.raises(ValueError, match="Preprocessor"):
            active_strategy.build_agent_resource("agent_a", definition, resources)

    def test_raises_on_invalid_preprocessor_entrypoint(self) -> None:
        broken_def = {
            "agents": {
                "agent_a": {
                    "rules": {},
                    "pre_processing": {"source": {"entrypoint": "no-dot"}},
                }
            }
        }
        resources = [("agent_a:preprocessor_folder", b"pre")]
        with pytest.raises(ValueError, match="Invalid preprocessor entrypoint"):
            active_strategy.build_agent_resource("agent_a", broken_def, resources)


class TestBuildActiveTestEvent:
    def test_injects_jwt_when_missing(self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        mocker.patch("app.services.runs.token_issuer.generate_jwt_token", return_value="abc.def.ghi")

        event = active_strategy.build_active_test_event(
            test_data={
                "payload": {"x": 1},
                "params": {"channel": "whatsapp"},
                "credentials": {"k": "v"},
                "project": {"uuid": "p", "vtex_account": "loja"},
            },
            token_issuer=_active_issuer(),
        )

        assert event["payload"] == {"x": 1}
        assert event["params"] == {"channel": "whatsapp"}
        assert event["credentials"] == {"k": "v"}
        assert event["project"]["auth_token"] == "abc.def.ghi"
        assert event["project"]["vtex_account"] == "loja"
        assert event["project_rules"] == []
        assert event["ignored_official_rules"] == []
        assert event["global_rule"] is None
        minted = [record for record in caplog.records if record.message.startswith("event=run_token_minted")]
        assert len(minted) == 1

    def test_preserves_existing_auth_token(self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        spy = mocker.patch("app.services.runs.token_issuer.generate_jwt_token", return_value="should-not-be-used")
        event = active_strategy.build_active_test_event(
            test_data={"project": {"auth_token": "user-token"}},
            token_issuer=_active_issuer(),
        )
        assert event["project"]["auth_token"] == "user-token"
        spy.assert_not_called()
        assert not any(record.message.startswith("event=run_token_minted") for record in caplog.records)

    def test_falls_back_to_credentials_argument(self, mocker: MockerFixture) -> None:
        mocker.patch("app.services.runs.token_issuer.generate_jwt_token", return_value="t")
        event = active_strategy.build_active_test_event(
            test_data={"project": {}},
            token_issuer=_active_issuer(),
            fallback_credentials={"api_key": "secret"},
        )
        assert event["credentials"] == {"api_key": "secret"}

    def test_parses_string_project_field(self, mocker: MockerFixture) -> None:
        mocker.patch("app.services.runs.token_issuer.generate_jwt_token", return_value="t")
        event = active_strategy.build_active_test_event(
            test_data={"project": json.dumps({"foo": "bar"})},
            token_issuer=_active_issuer(),
        )
        assert event["project"]["foo"] == "bar"
