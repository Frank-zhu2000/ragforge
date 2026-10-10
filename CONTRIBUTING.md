# Contributing to RAGForge

Thank you for your interest in contributing to **RAGForge**! 🎉

We welcome contributions of all kinds: bug fixes, new chunkers, performance optimizations, documentation improvements, and evaluation benchmarks.

---

## Development Setup

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/Frank-zhu2000/ragforge.git
   cd ragforge
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install in Editable Mode**:
   ```bash
   pip install -e ".[dev]"
   ```

---

## Running Tests

We maintain strict test coverage standards. Before opening a PR, ensure all tests pass:

```bash
# Run the complete test suite
python -m unittest discover -s tests -v

# Run examples to ensure no API regressions
python examples/01_quickstart.py
python examples/02_hybrid_search.py
python examples/03_pipeline_eval.py
```

---

## Coding Guidelines

- Follow PEP 8 style guidelines.
- Add docstrings to all public classes and methods.
- Ensure all new algorithms have corresponding tests in `tests/`.
- Keep the core library lightweight and free of heavy mandatory dependencies.

---

## Pull Request Process

1. Create a descriptive branch: `git checkout -b feat/my-new-chunker` or `fix/bm25-tokenizer-cjk`.
2. Commit changes using standard conventional commit messages (`feat: ...`, `fix: ...`, `docs: ...`).
3. Push to your fork and submit a Pull Request following the PR template.
4. All CI checks must pass before merging.
