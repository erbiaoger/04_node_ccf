"""Compatibility imports for the split, masked node SAC reader.

The reader now returns sample×station validity, includes SAC b and uses
half-open sample intervals. Implementation and usage: node_io/README.md.
"""

try:
    from .node_io import (
        NodeCatalog,
        NodeSACReader,
        SACSegment,
        StationRecord,
        load_node_catalog,
    )
except ImportError:
    from node_io import (
        NodeCatalog,
        NodeSACReader,
        SACSegment,
        StationRecord,
        load_node_catalog,
    )

__all__ = [
    "NodeCatalog",
    "NodeSACReader",
    "SACSegment",
    "StationRecord",
    "load_node_catalog",
]
