# ezgatr Tests README

## Overview

This directory contains test cases for the `ezgatr` library, ensuring that its components function correctly and as expected. Tests are organized to mirror the sub-module structure of the main package.

## Test Organization

- **Sub-module Specific Tests**: Tests for specific sub-modules (e.g., `ezgatr.nn`) reside within corresponding directories under `tests` (e.g., `tests.nn`).

- **Thematic Tests**: Tests covering specific themes that do not belong to any particular module are placed in the `tests/thematic` directory. For example, `tests.thematic.test_regress_with_clifford.py` ensures that the library's geometric algebra operations align with those of the `clifford` package.

## Configuration Management

To avoid hard-coding configurations and to maintain consistency across tests, configurations are centralized into dedicated files.

- **Core Configurations**: Shared configurations, such as execution devices, numerical precision tolerance, and test data generation parameters, are defined in `tests/config/core.toml`.

- **Module/Theme-Specific Configurations**: Configurations unique to specific sub-modules or themes are organized in respective files within the `tests/config` directory:
  - `interfaces.toml`
  - `thematic.toml`

## File Structure

```
tests/
├── __init__.py
├── conftest.py
├── utils.py
├── config/
│   ├── __init__.py
│   ├── core.toml
│   ├── interfaces.toml
│   └── thematic.toml
├── interfaces/
│   ├── __init__.py
│   ├── test_plane.py
│   ├── test_point.py
│   └── ...
├── nn/
│   ├── __init__.py
│   ├── test_functional.py
│   └── test_modules.py
└── thematic/
    ├── __init__.py
    ├── README.md
    ├── test_gradient_flow.py
    ├── test_operator_equivariance.py
    └── test_regress_with_clifford.py
```

## Forward

This README provides an introduction to how tests and configurations are organized in the `tests` directory. Future updates will provide more detailed documentation as the test suite expands.
