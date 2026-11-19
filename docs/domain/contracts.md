# Mhyra Vision Domain Documentation

## Overview

The `domain/` package forms the pure Clean Architecture core of the **Mhyra Vision** Edge AI platform. It encapsulates business entities, value objects, domain contract interfaces, event payloads, and platform exceptions with zero external framework dependencies (no PyTorch, OpenCV, TensorRT, or NumPy imports).

---

## Package Structure

```
domain/
├── entities/           # Core immutable value objects & entities
│   ├── __init__.py
│   └── frame.py        # Frame, BoundingBox, Point, Embedding, Pose, TextRegion, TrackTarget
├── contracts/          # Abstract interface contracts
│   ├── __init__.py
│   ├── skill.py        # ISkillContract
│   ├── runtime.py      # IRuntimeContract
│   ├── rule.py         # IRuleContract
│   └── sink.py         # ISinkContract
├── events/             # Domain event payloads
│   ├── __init__.py
│   ├── base.py         # DomainEvent
│   └── detection.py    # DetectionEvent, RuleViolationEvent, SystemAlertEvent
└── exceptions/         # Domain exception hierarchy
    ├── __init__.py
    └── base.py         # DomainException, DomainValidationError, InvalidEntityStateError
```

---

## Core Principles

1. **Pure Python**: Implemented using pure Python 3.10 standard library capabilities (`dataclasses`, `typing`, `abc`, `enum`, `uuid`, `datetime`).
2. **Immutability**: Value objects are enforced as immutable via `@dataclass(frozen=True, slots=True)`.
3. **Invariant Enforcement**: Validation checks are executed inside `__post_init__` methods, raising `DomainValidationError` when invariants are violated.
4. **Strict Typing**: 100% type hint coverage clean under `mypy --strict`.
