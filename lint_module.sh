echo "Running ruff on project"
poetry run ruff check
echo "Running mypy on project"
poetry run mypy . --config-file ./pyproject.toml --namespace-packages
