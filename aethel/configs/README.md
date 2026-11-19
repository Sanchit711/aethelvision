# Aurelix Configs Submodule (`aurelix.configs`)

## Purpose
The **Configs Submodule** manages declarative YAML-based configuration schemas, validation rules, and hierarchy merging across all framework subsystems.

## Responsibilities
- Provide `ConfigManager` for parsing, merging, and validating YAML files.
- Provide `ConfigDict` wrapper for dot-notation attribute access.
- Maintain versioned declarative YAML specs for datasets, models, training runs, deployment builds, and experiments.

## Public API
- `ConfigManager`: Config parsing and merging utility.
- `ConfigDict`: Attribute-accessible configuration dictionary.

## Dependencies
- `pyyaml`, `pathlib`
- `aurelix.utils.exceptions.ConfigurationException`

## Future Work
- Add Pydantic schema validation models for strict type checking over YAML inputs.
