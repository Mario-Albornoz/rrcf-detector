"""Partitioning package for distributing work across processes."""

from .strategy import hash_based_partitioner

__all__ = ["hash_based_partitioner"]
