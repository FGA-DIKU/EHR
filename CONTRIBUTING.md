# Contributing Guidelines

Thank you for considering contributing to our project! Please follow the guidelines below to ensure consistency and quality.

---

## Code Style & Quality

- **Formatting:** Use **Black** (`≥25.1.0`) for code formatting.
- **Linting:** Use **Flake8** for code quality checks.
- **Typing:** Add type hints for function parameters and return values.
- **Structure:**  
  - Follow the project directory structure.  
  - Keep functions small and focused (single responsibility).  
  - Use constants for column names and special values (`corebehrt/constants/`).  
  - Centralize configuration (`corebehrt/configs/`).  
- **Error Handling:**  
  - Implement proper exception handling.  
  - Validate inputs and provide descriptive error messages.  
- **Testing:**  
  - Write unit tests using `unittest` and place them in `tests/` mirroring the package structure.  
  - Run tests locally before submitting a PR:

  ```bash
  python -m unittest discover -s tests
  ```

## Pull Request Process

1. Fork the repository and create a feature branch (`git checkout -b feature/amazing-feature`)
2. Make your changes and ensure formatting and linting pass:

   ```bash
   black corebehrt tests
   flake8 corebehrt tests --select=E9,F63,F7,F82,U100,E711,E712,E713,E714,E721,F401,F402,F405,F811,F821,F822,F823,F831,F841,F901,
   ```

3. Commit using conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for adding tests
   - `refactor:` for code refactoring

4. Push to your fork and open a Pull Request

### Handling Data Changes in Tests

When making changes that affect output data, follow these steps:

1. If the test fails due to intentional changes in output data:

   ```bash
   # 1. Remove existing test data
   rm -rf tests/data/features/
   rm -rf tests/data/outcomes/

   # 2. Regenerate features test data
   python -m corebehrt.main.create_data \
     --config tests/pipeline_configs/create_data.yaml

   # 3. Regenerate outcomes test data
   python -m corebehrt.main.create_outcomes \
     --config tests/pipeline_configs/create_outcomes.yaml
   ```

2. The test configs must use the same paths:
   - Features test data: `tests/data/features/`
   - Outcomes test data: `tests/data/outcomes/`
   - Test configs: `tests/pipeline_configs/*.yaml`

3. Verify the changes:

   ```bash
   python -m unittest discover -s tests/test_main
   ```

Note: Always document significant changes to the output data format in the PR description.

### PR Requirements

- Follow existing code style
- Add tests for new features
- Update documentation as neede
- All CI checks must pass
- Keep changes focused and atomic
- **Important:** If you add new scripts or change input/output paths, update [azure components](corebehrt/azure/components) and [azure main](corebehrt/azure/__main__.py) accordingly.

Thank you for contributing! 🚀
