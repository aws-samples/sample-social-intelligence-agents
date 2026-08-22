"""Tests for the synth gate in scripts/check_synth.py.

The gate exists to catch a stack that lost a cdk-nag suppression, so these tests pin
both directions: a clean assembly passes, and each way the assembly can be wrong
produces a failure. Without the negative cases a gate that never runs is
indistinguishable from one that found nothing.
"""

import csv
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import check_synth  # noqa: E402  (path is set up above so the script is importable)

NAG_COLUMNS = ["Rule ID", "Resource ID", "Compliance", "Exception Reason", "Rule Level", "Rule Info"]


def _write_assembly(
    directory: pathlib.Path,
    *,
    resources: dict | None = None,
    nag_rows: list[dict[str, str]] | None,
) -> pathlib.Path:
    """Build a minimal cloud assembly on disk.

    Args:
        directory: Directory to populate.
        resources: Template resources. ``None`` writes no template at all.
        nag_rows: cdk-nag report rows. ``None`` writes no report at all.

    Returns:
        The assembly directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if resources is not None:
        (directory / "TestStack.template.json").write_text(json.dumps({"Resources": resources}))
    if nag_rows is not None:
        with (directory / "AwsSolutions--TestStack-NagReport.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=NAG_COLUMNS)
            writer.writeheader()
            writer.writerows(nag_rows)
    return directory


def _row(compliance: str, rule_id: str = "AwsSolutions-IAM5") -> dict[str, str]:
    """Return one cdk-nag report row with the given compliance status."""
    return {
        "Rule ID": rule_id,
        "Resource ID": "TestStack/Thing/Resource",
        "Compliance": compliance,
        "Exception Reason": "",
        "Rule Level": "Error",
        "Rule Info": "info",
    }


def test_clean_assembly_passes(tmp_path):
    assembly = _write_assembly(
        tmp_path / "clean",
        resources={"Thing": {"Type": "AWS::S3::Bucket"}},
        nag_rows=[_row("Compliant"), _row("Suppressed")],
    )
    assert check_synth.check(assembly) == []


def test_non_compliant_row_fails(tmp_path):
    assembly = _write_assembly(
        tmp_path / "dirty",
        resources={"Thing": {"Type": "AWS::S3::Bucket"}},
        nag_rows=[_row("Compliant"), _row("Non-Compliant", "AwsSolutions-APIG3")],
    )
    failures = check_synth.check(assembly)
    assert len(failures) == 1
    assert "AwsSolutions-APIG3" in failures[0]
    assert "Non-Compliant" in failures[0]


def test_missing_nag_report_fails(tmp_path):
    """A report-less assembly must fail: cdk-nag not running is not a pass."""
    assembly = _write_assembly(tmp_path / "noreport", resources={"Thing": {"Type": "AWS::S3::Bucket"}}, nag_rows=None)
    failures = check_synth.check(assembly)
    assert len(failures) == 1
    assert "cdk-nag did not run" in failures[0]


def test_empty_nag_report_fails(tmp_path):
    """A header-only report means zero rules were evaluated."""
    assembly = _write_assembly(tmp_path / "emptyreport", resources={"Thing": {"Type": "AWS::S3::Bucket"}}, nag_rows=[])
    assert any("cdk-nag did not run" in failure for failure in check_synth.check(assembly))


def test_missing_template_fails(tmp_path):
    assembly = _write_assembly(tmp_path / "notemplate", resources=None, nag_rows=[_row("Compliant")])
    assert any("no *.template.json" in failure for failure in check_synth.check(assembly))


def test_resourceless_stack_fails(tmp_path):
    assembly = _write_assembly(tmp_path / "empty", resources={}, nag_rows=[_row("Compliant")])
    assert any("0 resources" in failure for failure in check_synth.check(assembly))


def test_absent_directory_fails(tmp_path):
    failures = check_synth.check(tmp_path / "never-synthesized")
    assert len(failures) == 1
    assert "did synth run?" in failures[0]


@pytest.mark.parametrize("statuses", [["Compliant"], ["Suppressed"], ["Compliant", "Suppressed"]])
def test_compliant_and_suppressed_are_both_acceptable(tmp_path, statuses):
    assembly = _write_assembly(
        tmp_path / ("ok-" + "-".join(statuses)),
        resources={"Thing": {"Type": "AWS::S3::Bucket"}},
        nag_rows=[_row(status) for status in statuses],
    )
    assert check_synth.check(assembly) == []
