# Contributing to HematoVision

Thank you for your interest! Here's how to contribute:

## Setup

```bash
git clone https://github.com/your-username/hematovision.git
cd hematovision
pip install -e ".[dev]"
```

## Workflow

1. Fork the repo and create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes with tests
3. Run `pytest` to ensure all tests pass
4. Run `black src/ tests/` and `isort src/ tests/` for formatting
5. Open a Pull Request with a clear description

## Areas to Contribute

- New backbone integrations via timm
- Additional augmentation strategies
- Multi-GPU / DDP training support
- ONNX / TorchScript export
- Better stain normalization methods
- Dataset support for other blood cell types
