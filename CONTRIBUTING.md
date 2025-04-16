# Contributing Guidelines

Thank you for considering contributing to our project! Please follow the guidelines below to ensure consistency and quality.

---

## Code Style & Quality

- **Formatting:** Use **ruff** for code formatting.
  - Run `ruff format corebehrt tests` to format the code.
- **Linting:** Use **ruff** for code quality checks.
  - Run `ruff check corebehrt tests --select E9,F63,F7,F82,E711,E712,E713,E714,E721,F401,F402,F405,F811,F821,F822,F823,F841,F901` to check linting.
- **Docstrings:** Use **docstr-coverage** to check docstring coverage.
  - Run `docstr-coverage corebehrt --skip-magic --skip-init` to check docstring coverage.
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

   You can generate a coverage report by running:

   ```bash
   coverage run -m unittest discover -s tests
   coverage report
   ```

## Pull Request Process

1. Fork the repository and create a feature branch (`git checkout -b feature/amazing-feature`)
2. Make your changes and ensure formatting and linting pass:

   ```bash
   ruff format --check corebehrt tests
   ruff check corebehrt tests --select E9,F63,F7,F82,E711,E712,E713,E714,E721,F401,F402,F405,F811,F821,F822,F823,F841,F901
   ```

3. Commit using conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for adding tests
   - `refactor:` for code refactoring

4. Push to your fork and open a Pull Request

### Updating Test Data

If a test fails due to intentional changes in output data (e.g., changes in feature calculation or data processing), you'll need to regenerate the test data.

#### Steps to Update

1. **Regenerate Test Data**

   Run the update script from the project root directory:

   ```bash
   # First time only: make script executable
   chmod +x tests/data/update.sh

   # Run the update script
   ./tests/data/update.sh
   ```

   This script will:
   - Remove existing test data
   - Regenerate features test data
   - Regenerate outcomes test data

2. **Verify Changes**

   Run the tests to ensure everything works:

   ```bash
   python -m unittest discover -s tests/test_main
   ```

### PR Requirements

- Follow existing code style
- Add tests for new features
- Update documentation as neede
- All CI checks must pass
- Keep changes focused and atomic
- **Important:** If you add new scripts or change input/output paths, update [azure components](corebehrt/azure/components) and [azure main](corebehrt/azure/__main__.py) accordingly.

Thank you for contributing! 🚀
