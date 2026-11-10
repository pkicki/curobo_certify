"""Certification status vocabulary and conservative aggregation."""
from enum import Enum

class Status(str, Enum):
    CERTIFIED="CERTIFIED"; VIOLATED="VIOLATED"; UNKNOWN_DEPTH="UNKNOWN_DEPTH"
    UNKNOWN_NODES="UNKNOWN_NODES"; UNKNOWN_TIME="UNKNOWN_TIME"; UNSUPPORTED="UNSUPPORTED"
    NUMERICAL_ERROR="NUMERICAL_ERROR"; INVALID_INPUT="INVALID_INPUT"
    PARTIAL_K3="PARTIAL_K3"

def aggregate(requested):
    """Only all-CERTIFIED requested obligations aggregate to CERTIFIED."""
    values=list(requested.values())
    if values and all(v == Status.CERTIFIED or v == Status.CERTIFIED.value for v in values): return Status.CERTIFIED
    if any(v == Status.VIOLATED or v == Status.VIOLATED.value for v in values): return Status.VIOLATED
    if any(v == Status.UNSUPPORTED or v == Status.UNSUPPORTED.value for v in values): return Status.PARTIAL_K3
    return Status.UNKNOWN_DEPTH
