from sprint3_tools_mcp import (
    build_heliodesk_tool_registry,
    build_task_answer_signals,
    review_decision_note,
    review_recovery_plan,
    run_direct_authorization_check,
    run_failure_scenarios,
    run_scripted_tool_loop,
    validate_checkpoint_evidence,
)


def test_sprint_3_helpers_cover_the_guided_checkpoint_path() -> None:
    registry = build_heliodesk_tool_registry()

    authorization_result, authorization_payload = run_direct_authorization_check(registry)
    assert authorization_result.ok is True
    assert authorization_payload["authorization"]["evidence_id"] == "AUTH-8841"

    failure_evidence, _ = run_failure_scenarios(registry)
    recovery_review = review_recovery_plan(
        failure_evidence,
        {
            "invalid_input": "ask_for_missing_input",
            "malformed_response": "reject_and_log",
            "missing_data": "reject_and_log",
            "timeout": "retry_once_then_route",
            "empty_result": "honest_no_result",
            "permission_failure": "route_to_authorized_owner",
        },
    )
    assert all(row["passes"] for row in recovery_review)

    scripted_run, _ = run_scripted_tool_loop(registry)
    assert [result.name for result in scripted_run.tool_results] == [
        "check_export_authorization",
        "search_heliodesk_policy",
    ]

    decision_note_review = review_decision_note(
        "The direct authorization contract requires confirmed fields. "
        "A timeout or empty result is a failure signal, not approval. "
        "The MCP boundary returns policy evidence while the direct authorization "
        "tool returns current workspace state, so downstream logic must use both "
        "sources before it writes an answer."
    )
    assert decision_note_review["passes"]
    assert set(build_task_answer_signals()) == {
        "task_a_direct_tool_contract",
        "task_b_failure_handling",
        "task_c_multi_step_and_mcp",
    }

    validate_checkpoint_evidence(
        authorization_payload=authorization_payload,
        recovery_review=recovery_review,
        scripted_run=scripted_run,
        validated_mcp_rows=[{"index": 0, "score": 1.0, "text": "authorization evidence"}],
        decision_note_review=decision_note_review,
    )
