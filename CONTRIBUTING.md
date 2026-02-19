# Contributing to Playchitect

Thank you for your interest in contributing to Playchitect!

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- Virtual environment tool (venv)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/james-westwood/playchitect.git
cd playchitect

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest -v
```

## Development Workflow

### Feature Development

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Write failing tests first (TDD approach)
3. Implement the feature
4. Run pre-commit checks:
   ```bash
   pre-commit run --all-files
   ```

5. Run full test suite:
   ```bash
   pytest -v --cov=playchitect
   ```

6. Commit with descriptive message:
   ```bash
   git commit -m "feat(module): add new feature"
   ```

### Commit Message Convention

We follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

Closes #<issue>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or modifications
- `refactor`: Code refactoring
- `chore`: Build/tooling changes

**Example:**
```
feat(clustering): implement K-means with auto-K detection

- Add elbow method for optimal K determination
- Implement cluster splitting for target length
- Add unit tests with synthetic data

Closes #11
```

## Code Quality Standards

### Pre-commit Hooks

All commits must pass these checks:

1. **black** - Code formatting (line-length=100)
2. **flake8** - Style linting
3. **mypy** - Type checking (strict mode)
4. **pytest** - Unit tests for modified modules
5. **GUI smoke tests** - Layout and accessibility checks (when applicable)

### Test Coverage

- Minimum 85% code coverage for core modules
- All new features must include tests
- Use pytest fixtures for common test setup

### Type Hints

All functions must include type hints:

```python
def process_track(filepath: Path, bpm: float) -> TrackMetadata:
    """Process audio track and extract metadata."""
    ...
```

## Testing

### Running Tests

```bash
# All tests
pytest -v

# Specific test file
pytest tests/unit/test_audio_scanner.py -v

# With coverage report
pytest --cov=playchitect --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Structure

- `tests/unit/` - Unit tests for individual modules
- `tests/integration/` - End-to-end workflow tests
- `tests/gui/` - GUI smoke tests

### Writing Tests

```python
import pytest
from pathlib import Path
from playchitect.core.audio_scanner import AudioScanner

class TestAudioScanner:
    def test_scan_directory(self, tmp_path):
        """Test scanning directory for audio files."""
        scanner = AudioScanner()
        (tmp_path / "track.mp3").touch()

        result = scanner.scan(tmp_path)

        assert len(result) == 1
        assert result[0].name == "track.mp3"
```

## GUI Development

### GTK4 + libadwaita Guidelines

- Follow [GNOME Human Interface Guidelines](https://developer.gnome.org/hig/)
- Use libadwaita widgets for native GNOME appearance
- Support adaptive layouts for different screen sizes
- Include accessibility attributes (ARIA labels)
- Test with keyboard navigation

### GUI Testing

GUI smoke tests verify:
- All required widgets render correctly
- GNOME HIG compliance (spacing, alignment)
- Keyboard navigation functionality
- Accessibility attributes

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def analyze_intensity(filepath: Path, sample_rate: int = 22050) -> IntensityVector:
    """
    Analyze audio intensity using librosa.

    Args:
        filepath: Path to audio file
        sample_rate: Target sample rate for analysis (default: 22050)

    Returns:
        IntensityVector with normalized features

    Raises:
        FileNotFoundError: If audio file does not exist
        ValueError: If sample rate is invalid
    """
    ...
```

### README Updates

Update README.md when adding:
- New features
- CLI commands
- Configuration options

## Issue Tracking

### Creating Issues

Use issue templates:
- Bug reports: Include steps to reproduce
- Feature requests: Describe use case and expected behavior
- Questions: Provide context

### Labels

- `priority-critical` / `priority-high` / `priority-medium` / `priority-low`
- `type-feature` / `type-bug` / `type-docs` / `type-test`
- `area-core` / `area-cli` / `area-gui` / `area-analysis`
- `good-first-issue` / `needs-testing` / `needs-review`

## Pull Requests

### Before Submitting

- [ ] All tests pass (`pytest -v`)
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] Code coverage >85% for new code
- [ ] Documentation updated
- [ ] CHANGELOG updated (if applicable)

### PR Template

```markdown
## Description
Brief description of changes

## Related Issues
Closes #123

## Changes Made
- List of changes
- ...

## Testing
How changes were tested

## Screenshots (if applicable)
For GUI changes
```

## Questions?

- Open an issue with the `question` label
- Check existing documentation
- Review closed issues for similar questions

## License

By contributing, you agree that your contributions will be licensed under the GPL-3.0 license.
