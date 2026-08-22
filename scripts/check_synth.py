"""Assert a synthesized CDK cloud assembly is complete and cdk-nag clean.

CI runs this after synthesizing the app so infrastructure changes are validated the
same way application changes are. It exists because neither half of the obvious check
is trustworthy on its own:

- ``cdk synth`` exits 0 even when cdk-nag reports an unsuppressed **warning**, so exit
  status alone silently passes a stack that lost a suppression.
- "no findings" and "cdk-nag never ran" produce identical silence, so the absence of
  findings is only meaningful alongside a count of what was actually evaluated.

So this asserts on positive evidence: a template with resources, a cdk-nag report with
rows, and zero non-compliant rows among them.

Usage:
    CDK_OUTDIR=cdk.out python infra/app.py
    python scripts/check_synth.py cdk.out
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import sys

logger = logging.getLogger("check_synth")

# cdk-nag writes one report per pack per stack, named "<PackName>-<StackName>-NagReport.csv".
NAG_REPORT_GLOB = "*NagReport.csv"
COMPLIANCE_COLUMN = "Compliance"
NON_COMPLIANT = "Non-Compliant"


def _load_templates(assembly: pathlib.Path) -> dict[str, dict]:
    """Return every CloudFormation template in the assembly, keyed by file stem."""
    templates = {}
    for path in sorted(assembly.glob("*.template.json")):
        templates[path.name.removesuffix(".template.json")] = json.loads(path.read_text())
    return templates


def _load_nag_rows(assembly: pathlib.Path) -> list[dict[str, str]]:
    """Return every row across every cdk-nag CSV report in the assembly."""
    rows: list[dict[str, str]] = []
    for path in sorted(assembly.glob(NAG_REPORT_GLOB)):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def check(assembly: pathlib.Path) -> list[str]:
    """Validate a synthesized assembly, returning a list of failure messages.

    Args:
        assembly: Cloud assembly directory produced by ``app.synth()``.

    Returns:
        Human-readable failures. Empty means the assembly passed.
    """
    failures: list[str] = []

    if not assembly.is_dir():
        return [f"assembly directory {assembly} does not exist — did synth run?"]

    templates = _load_templates(assembly)
    if not templates:
        failures.append(f"no *.template.json in {assembly} — synth produced no stacks")
    for name, template in templates.items():
        resources = template.get("Resources") or {}
        if not resources:
            failures.append(f"stack {name} synthesized 0 resources")
        else:
            logger.info("stack %s: %d resources", name, len(resources))

    nag_rows = _load_nag_rows(assembly)
    if not nag_rows:
        failures.append(
            f"no {NAG_REPORT_GLOB} rows in {assembly} — cdk-nag did not run, so "
            "'no findings' proves nothing. Check the pack is registered in infra/app.py."
        )
    else:
        counts: dict[str, int] = {}
        for row in nag_rows:
            status = row.get(COMPLIANCE_COLUMN, "?")
            counts[status] = counts.get(status, 0) + 1
        logger.info(
            "cdk-nag: %d rules evaluated (%s)", len(nag_rows), ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        )
        for row in nag_rows:
            if row.get(COMPLIANCE_COLUMN) == NON_COMPLIANT:
                failures.append(
                    f"cdk-nag {row.get('Rule Level', '?')} {row.get('Rule ID', '?')} "
                    f"on {row.get('Resource ID', '?')} is {NON_COMPLIANT}"
                )

    return failures


def main() -> int:
    """Entry point. Returns a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("assembly", nargs="?", default="cdk.out", help="cloud assembly directory (default: cdk.out)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    failures = check(pathlib.Path(args.assembly))
    if failures:
        for failure in failures:
            logger.error("FAIL %s", failure)
        return 1
    logger.info("OK synthesized assembly is complete and cdk-nag clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
