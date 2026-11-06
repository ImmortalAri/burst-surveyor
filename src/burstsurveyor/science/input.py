"""Read only raw intensities and acquisition grid scalars from a bounded NPZ."""

import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import pickletools
import re
import zipfile
import zlib

import numpy as np

from .engine import InputError

MAX_MEMBER = 160 * 1024**2
MAX_METADATA = 65536
GRID_KEYS = ("res_time", "res_freq", "freqs_bin0", "times_bin0", "is_dedispersed")


class _LegacyDtype:
    """Decode only a bounded primitive dtype, without invoking NumPy pickle reconstruction."""

    def __init__(self, code, *_):
        if not isinstance(code, str) or not re.fullmatch(
            r"[<>=|]?[biufcOSU]\d{0,5}", code
        ):
            raise InputError("Unsupported legacy metadata dtype")
        self.dtype = np.dtype(code)
        if self.dtype.itemsize > MAX_METADATA:
            raise InputError("Legacy metadata item is too large")

    def __setstate__(self, state):
        if (
            not isinstance(state, tuple)
            or len(state) < 2
            or state[1] not in ("<", ">", "=", "|")
        ):
            raise InputError("Invalid legacy dtype byte order")
        self.dtype = self.dtype.newbyteorder(state[1])


class _LegacyArray:
    """An inert stand-in for the outer metadata container. No array allocation occurs."""

    def __setstate__(self, state):
        if not isinstance(state, tuple) or len(state) != 5:
            raise InputError("Unsupported legacy metadata container")
        _, shape, dtype, _, contents = state
        if not isinstance(dtype, _LegacyDtype):
            raise InputError("Invalid legacy array dtype")
        self.metadata = None
        if (
            shape == ()
            and dtype.dtype.kind == "O"
            and isinstance(contents, list)
            and len(contents) == 1
        ):
            self.metadata = contents[0]


def _legacy_reconstruct(kind, shape, marker):
    if kind is not _LegacyArray or shape != (0,) or marker != b"b":
        raise InputError("Unsupported legacy array reconstruction")
    return _LegacyArray()


def _legacy_scalar(descriptor, data):
    if not isinstance(descriptor, _LegacyDtype) or descriptor.dtype.kind not in "biuf":
        raise InputError("Unsupported legacy metadata scalar")
    if not isinstance(data, bytes) or len(data) != descriptor.dtype.itemsize:
        raise InputError("Invalid legacy scalar length")
    return np.frombuffer(data, dtype=descriptor.dtype, count=1)[0].item()


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("numpy", "ndarray"): _LegacyArray,
            ("numpy", "dtype"): _LegacyDtype,
            ("numpy._core.multiarray", "_reconstruct"): _legacy_reconstruct,
            ("numpy._core.multiarray", "scalar"): _legacy_scalar,
        }
        try:
            return allowed[(module.replace("numpy.core.", "numpy._core."), name)]
        except KeyError:
            raise InputError(
                "Unsupported global in legacy acquisition metadata"
            ) from None


def _header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream)
    raise InputError(f"Unsupported NPY format version {version}")


def _metadata(payload):
    stream = io.BytesIO(payload)
    shape, _, dtype = _header(stream)
    if shape != ():
        raise InputError("Expected a scalar metadata.npy container")
    if dtype.kind in "SU":
        if dtype.itemsize != len(payload) - stream.tell():
            raise InputError("Metadata string length disagrees with its NPY header")
        text = np.frombuffer(stream.read(), dtype=dtype, count=1)[0].item()
        value = json.loads(text)
    elif dtype.kind == "O":
        body = stream.read()
        for opcode, arg, _ in pickletools.genops(body):
            if opcode.name in {
                "EXT1",
                "EXT2",
                "EXT4",
                "PERSID",
                "BINPERSID",
                "NEXT_BUFFER",
                "READONLY_BUFFER",
            }:
                raise InputError("Unsupported reference in legacy metadata")
            if opcode.name in {"LONG_BINPUT", "PUT"} and int(arg) > MAX_METADATA:
                raise InputError("Oversized legacy metadata memo")
        outer = _MetadataUnpickler(io.BytesIO(body)).load()
        value = outer.metadata if isinstance(outer, _LegacyArray) else None
    else:
        raise InputError(
            "Expected scalar JSON text or legacy scalar-object metadata.npy"
        )
    if not isinstance(value, dict):
        raise InputError("Acquisition metadata must be a mapping")
    return value


def _inspect(archive, size):
    for name in ("data_full.npy", "metadata.npy"):
        if archive.namelist().count(name) != 1:
            raise InputError(f"Missing or duplicate NPZ member {name}")
    if archive.getinfo("metadata.npy").file_size > MAX_METADATA:
        raise InputError("Acquisition metadata exceeds 64 KiB")
    expanded = archive.getinfo("data_full.npy").file_size
    if expanded > MAX_MEMBER:
        raise InputError("Expanded intensity array exceeds 160 MiB")
    with archive.open("data_full.npy") as stream:
        shape, _, dtype = _header(stream)
        if len(shape) != 2 or min(shape) <= 20 or dtype.kind not in "fiu":
            raise InputError(
                "Expected a numeric frequency-by-time array with both dimensions above 20"
            )
        if stream.tell() + math.prod(shape) * dtype.itemsize != expanded:
            raise InputError("Intensity payload length disagrees with its NPY header")
    meta = _metadata(archive.read("metadata.npy"))
    grid = {}
    for key in GRID_KEYS[:-1]:
        value = meta.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise InputError(f"Missing or invalid acquisition scalar: {key}")
        grid[key] = float(value)
    dedispersed = meta.get("is_dedispersed")
    if type(dedispersed) not in (bool, int) or dedispersed != 1:
        raise InputError("Only dedispersed input with is_dedispersed=true is supported")
    grid["is_dedispersed"] = True
    if (meta.get("num_freq", shape[0]), meta.get("num_time", shape[1])) != shape:
        raise InputError("Metadata grid dimensions differ from the intensity array")
    if grid["res_time"] <= 0 or grid["res_freq"] == 0:
        raise InputError("Invalid time or frequency spacing")
    endpoints = [
        grid["freqs_bin0"] + (i + 0.5) * grid["res_freq"] for i in (0, shape[0] - 1)
    ]
    if min(endpoints) <= 0 or not all(math.isfinite(x) for x in endpoints):
        raise InputError("Invalid frequency centres")
    return dict(
        shape=list(shape),
        grid=grid,
        dtype=str(dtype),
        expanded_bytes=expanded,
        input_bytes=size,
    )


_FORMAT_ERRORS = (
    ValueError,
    TypeError,
    OverflowError,
    EOFError,
    zipfile.BadZipFile,
    pickle.PickleError,
    IndexError,
    AttributeError,
    KeyError,
    RuntimeError,
    zlib.error,
)


def inspect_npz(path):
    """Inspect dimensions, acquisition scalars and expanded size without loading intensity pixels."""
    path = Path(path)
    size = path.stat().st_size
    if size > MAX_MEMBER:
        raise InputError("NPZ exceeds 160 MiB")
    try:
        with zipfile.ZipFile(path) as archive:
            return _inspect(archive, size)
    except InputError:
        raise
    except _FORMAT_ERRORS as exc:
        raise InputError(f"Invalid NPZ input: {exc}") from exc


def load_raw(path, scratch_directory):
    """Read pixels and metadata from the same original bytes that are hashed in the receipt."""
    path = Path(path)
    size = path.stat().st_size
    if size > MAX_MEMBER:
        raise InputError("NPZ exceeds 160 MiB")
    with path.open("rb") as stream:
        payload = stream.read(size + 1)
    if len(payload) != size:
        raise InputError("NPZ size changed during staging")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entry = _inspect(archive, len(payload))
            with archive.open("data_full.npy") as raw_file:
                raw = np.lib.format.read_array(raw_file, allow_pickle=False)
        entry["source_sha256"] = hashlib.sha256(payload).hexdigest()
        return raw, entry
    except InputError:
        raise
    except _FORMAT_ERRORS as exc:
        raise InputError(f"Invalid NPZ input: {exc}") from exc
