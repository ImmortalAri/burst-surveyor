"""Automated fitting of radio-burst dynamic spectra."""

__version__ = "0.3.1"


def fit(inputs, output, *, workers=3, resume=False, mode="bounded", exporter=None):
    """Fit one NPZ or a directory and return the verified batch summary."""
    from .batch import fit as batch_fit

    return batch_fit(inputs, output, workers=workers, resume=resume, mode=mode, exporter=exporter)


def verify(output):
    """Verify a completed batch and return its summary."""
    from .batch import verify as verify_batch

    return verify_batch(output)
