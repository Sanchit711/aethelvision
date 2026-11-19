"""Helper Utilities for Aethel Detector Architecture.

Responsibility:
Provides channel scaling functions, weight initialization, and tensor math helpers.
"""

def make_divisible(value: float, divisor: int = 8) -> int:
    """Ensure channel counts are divisible by divisor for hardware alignment."""
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value
