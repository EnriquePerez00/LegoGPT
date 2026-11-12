"""
GeometryCache: Singleton accessor for LDrawGeometryResolver.

Provides a cached, thread-safe instance of the geometry resolver
to avoid re-parsing the entire LDRAW database multiple times.
"""

from typing import Optional
from src.ldraw_geometry import LDrawGeometryResolver

_INSTANCE: Optional[LDrawGeometryResolver] = None

class GeometryCache:
    """Singleton wrapper for LDrawGeometryResolver."""
    
    @staticmethod
    def get() -> LDrawGeometryResolver:
        """Get or create the global GeometryCache instance."""
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = LDrawGeometryResolver()
        return _INSTANCE
    
    @staticmethod
    def reset():
        """Reset the cache (for testing)."""
        global _INSTANCE
        _INSTANCE = None

