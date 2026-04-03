"""Tests for por schema CLI command."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from proof_of_replica.cli.main import app
from proof_of_replica.core.schema import generate_json_schema


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestSchemaCommand:
    def test_schema_stdout(self, runner):
        result = runner.invoke(app, ["schema"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "properties" in data
        assert data.get("title") == "Profile"

    def test_schema_to_file(self, runner, tmp_path):
        output = tmp_path / "schema.json"
        result = runner.invoke(app, ["schema", "-o", str(output)])
        assert result.exit_code == 0
        assert output.exists()
        assert "Schema written" in result.output

    def test_schema_matches_generate_json_schema(self, runner):
        result = runner.invoke(app, ["schema"])
        cli_schema = json.loads(result.output)
        expected = generate_json_schema()
        assert cli_schema == expected

    def test_shipped_schema_in_sync(self):
        """Verify the shipped profile.schema.json matches the current model."""
        schema_path = Path("src/proof_of_replica/profile.schema.json")
        if not schema_path.exists():
            pytest.skip("profile.schema.json not found")
        shipped = json.loads(schema_path.read_text())
        expected = generate_json_schema()
        assert shipped == expected
