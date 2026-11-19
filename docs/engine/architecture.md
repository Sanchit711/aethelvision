# Mhyra Vision Engine Infrastructure Kernel Documentation

## Overview

The `engine/` package is the core infrastructure kernel of **Mhyra Vision**. It provides foundational system services including configuration validation, component registration, dependency injection container, asynchronous event bus messaging, structured JSON logging, and hardware-agnostic memory pool abstractions.

---

## Package Architecture & Subsystems

```
engine/
├── __init__.py         # Package root exports
├── config.py           # Pydantic configuration schemas & YAML loader
├── container.py        # Thread-safe Dependency Injection (DI) Container
├── events.py           # Asynchronous topic-based Event Bus
├── logger.py           # Structured JSON Logger
├── memory.py           # Memory pool abstractions & Host memory manager
├── registry.py         # Category-based Component & Plugin Registry
└── utils.py            # Timer, formatters, UUID utilities
```

---

## Subsystem Details

### 1. Configuration Management (`engine/config.py`)
Provides immutable, typed Pydantic models for system, hardware, stream, skill, and pipeline configurations. Includes YAML configuration parsing via `load_yaml_config()`.

### 2. Component & Plugin Registry (`engine/registry.py`)
Global category-based component registry (`REGISTRY`). Facilitates dynamic module discovery via python decorators:
```python
@REGISTRY.register(category="skills", name="yolov8")
class YOLOv8Skill:
    ...
```

### 3. Dependency Injection Container (`engine/container.py`)
IoC Container supporting `SINGLETON`, `TRANSIENT`, and `INSTANCE` registration lifetimes with automatic constructor parameter resolution.

### 4. Asynchronous Event Bus (`engine/events.py`)
Non-blocking publish/subscribe message broker. Supports glob wildcard topic subscriptions (e.g. `detection.*`), subscriber error isolation, and performance metrics.

### 5. Structured Logging (`engine/logger.py`)
Thread-safe logger emitting JSON-formatted log lines with extra context key-values for enterprise log collection (Elasticsearch, Loki).

### 6. Memory Pool Abstractions (`engine/memory.py`)
Defines `IMemoryBuffer` and `IMemoryPool` interface contracts along with a reference host RAM pool (`HostMemoryPool`) for zero-copy memory lifecycle management.
