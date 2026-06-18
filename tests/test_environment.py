from pathlib import Path


def test_torch_backed_dev_environment_is_documented():
    repo_root = Path(__file__).resolve().parents[1]

    requirements_dev = (repo_root / "requirements-dev.txt").read_text()
    readme = (repo_root / "README.md").read_text()

    assert "-r requirements.txt" in requirements_dev
    assert "pytest" in requirements_dev
    assert "requirements-dev.txt" in readme
