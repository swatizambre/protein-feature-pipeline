"""Typed errors for the pipeline (CLI / web / callers can branch on these)."""


class ProteinFeatureError(Exception):
    """Base error for this package."""


class InvalidPDBError(ProteinFeatureError):
    """PDB could not be parsed into usable protein residues."""


class SchemaError(ProteinFeatureError):
    """Encoded tensors / schema are missing keys or have wrong shapes."""


class ResourceLimitError(ProteinFeatureError):
    """Upload size, residue count, or other resource limit exceeded."""
