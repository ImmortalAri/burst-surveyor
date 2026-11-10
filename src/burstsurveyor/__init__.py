"""FitBurst reconstruction of radio-burst dynamic spectra."""

__version__ = "0.0.0"


def fit(inputs, output, *, workers=3, resume=False):
    """Fit one NPZ or a directory and return the verified batch summary."""
    from .batch import fit as batch_fit

    return batch_fit(inputs, output, workers=workers, resume=resume)


def verify(output):
    """Verify a completed batch and return its summary."""
    from .batch import verify as verify_batch

    return verify_batch(output)
