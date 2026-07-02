"""Sanity checks on repo layout -- catches an accidentally-deleted or
mis-pathed folder before someone discovers it mid-eval-run."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_DIRS = [
    "src",
    "src/comparison",
    "src/eval",
    "results/audio",
    "results/comparison",
    "results/eval",
    "results/logs",
    "reports",
]


def test_expected_directories_exist():
    for rel in EXPECTED_DIRS:
        assert (REPO_ROOT / rel).is_dir(), f"missing expected directory: {rel}"


def test_readme_and_license_present():
    assert (REPO_ROOT / "README.md").is_file()
    assert (REPO_ROOT / "LICENSE").is_file()


def test_reports_referenced_in_readme_exist():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for report in [
        "reports/tts_models_comparison.md",
        "reports/VoxCPM2_PoC_Infrastructure_Proposal.md",
        "reports/poc_rerun_results.md",
    ]:
        assert report in readme, f"{report} no longer referenced in README"
        assert (REPO_ROOT / report).is_file(), f"{report} referenced but missing on disk"
