from app.core.log_events import format_log_event


def test_empty_mapping_yields_only_the_event_name() -> None:
    assert format_log_event("ping", {}) == "event=ping"


def test_fields_follow_mapping_insertion_order() -> None:
    rendered = format_log_event(
        "project_mismatch_rejected",
        {
            "header_project_uuid": "header",
            "body_project_uuid": "body",
            "endpoint": "/api/v1/runs",
        },
    )

    assert rendered.startswith("event=project_mismatch_rejected ")
    assert rendered == (
        'event=project_mismatch_rejected header_project_uuid="header" '
        'body_project_uuid="body" endpoint="/api/v1/runs"'
    )


def test_none_field_is_omitted() -> None:
    rendered = format_log_event(
        "run_not_attributable",
        {"user_email": None, "request_id": "req-1", "endpoint": None},
    )

    assert rendered == 'event=run_not_attributable request_id="req-1"'
    assert "user_email" not in rendered
    assert "endpoint" not in rendered
    assert "null" not in rendered
    assert '""' not in rendered


def test_values_are_json_string_literals_on_one_line() -> None:
    rendered = format_log_event("audit", {"body": 'a"b\\c\n\r\t\x01'})

    assert rendered == 'event=audit body="a\\"b\\\\c\\n\\r\\t\\u0001"'
    assert "\n" not in rendered
    assert "\r" not in rendered


def test_quoted_value_cannot_inject_a_field() -> None:
    rendered = format_log_event("rejected", {"note": 'x" user_email="victim'})

    assert rendered == 'event=rejected note="x\\" user_email=\\"victim"'
    assert "\n" not in rendered
