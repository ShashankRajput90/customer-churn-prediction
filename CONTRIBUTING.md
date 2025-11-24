# Contributing to Customer Churn Prediction

Thank you for your interest in contributing! This guide will help you get started.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/ShashankRajput90/customer-churn-prediction/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots if applicable
   - Python version and OS

### Suggesting Enhancements

1. Open an issue with tag `enhancement`
2. Describe the feature and its benefits
3. Provide examples or mockups if possible

### Pull Requests

1. **Fork the repository**

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

2. **Create a feature branch**

```bash
git checkout -b feature/amazing-feature
```

3. **Make your changes**
   - Follow PEP 8 style guide
   - Add unit tests for new features
   - Update documentation

4. **Test your changes**

```bash
pytest tests/
pylint src/
black src/ --check
```

5. **Commit your changes**

```bash
git commit -m "Add amazing feature"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `style:` formatting, missing semi-colons, etc
- `refactor:` code restructuring
- `test:` adding tests
- `chore:` maintenance tasks

6. **Push to your fork**

```bash
git push origin feature/amazing-feature
```

7. **Open a Pull Request**
   - Describe your changes
   - Link related issues
   - Request review

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Virtual environment tool

### Local Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 pylint

# Run tests
pytest tests/ -v
```

## Code Style

### Python

- Follow [PEP 8](https://pep8.org/)
- Use `black` for formatting: `black src/`
- Use `flake8` for linting: `flake8 src/`
- Maximum line length: 100 characters

### Naming Conventions

- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private methods**: `_leading_underscore`

### Documentation

```python
def predict_churn(customer_data: pd.DataFrame) -> np.ndarray:
    """
    Predict churn probability for customers.
    
    Args:
        customer_data (pd.DataFrame): Customer features
        
    Returns:
        np.ndarray: Churn probabilities [0, 1]
        
    Example:
        >>> data = pd.DataFrame({'tenure': [12], 'MonthlyCharges': [70]})
        >>> probs = predict_churn(data)
        >>> print(probs[0])
        0.65
    """
    # Implementation
```

## Testing

### Unit Tests

```python
import pytest
from src.data_preprocessing import clean_data

def test_clean_data():
    """Test data cleaning removes missing values."""
    raw_data = load_test_data()
    cleaned = clean_data(raw_data)
    assert cleaned.isnull().sum().sum() == 0
```

### Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/test_data_preprocessing.py
```

## Project Structure

When adding new features:

- **Data processing**: `src/data_preprocessing.py`
- **Feature engineering**: `src/feature_engineering.py`
- **Model training**: `src/model_training.py`
- **Utilities**: `src/utils.py`
- **Tests**: `tests/test_*.py`

## Review Process

1. **Automated checks** run on every PR:
   - Unit tests must pass
   - Code coverage ≥80%
   - Linting passes
   
2. **Manual review**:
   - Code quality
   - Documentation completeness
   - Test coverage
   - Performance implications

3. **Approval**: At least one maintainer approval required

4. **Merge**: Squash and merge into main branch

## Areas for Contribution

### High Priority

- [ ] Add SHAP values for explainability
- [ ] Implement FastAPI endpoint
- [ ] Increase test coverage to 90%
- [ ] Add CI/CD pipeline (GitHub Actions)

### Medium Priority

- [ ] Improve dashboard UI/UX
- [ ] Add customer segmentation
- [ ] Implement time-series analysis
- [ ] Create Docker container

### Low Priority

- [ ] Add more visualizations
- [ ] Translate to other languages
- [ ] Create video tutorial
- [ ] Add example notebooks

## Questions?

Feel free to:
- Open an issue for discussion
- Start a discussion in [Discussions](https://github.com/ShashankRajput90/customer-churn-prediction/discussions)
- Reach out via email

---

**Thank you for contributing! 🚀**