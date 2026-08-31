"""Model package namespace for the standalone artifact.

Backends are imported directly from their subpackages by the eval scripts.
Avoid eager imports here so optional or removed families do not break the
public demos at package import time.
"""