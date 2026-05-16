import numpy as np

from bitarray import bitarray
from bitarray.util import int2ba, ba2int



def pack_to_bit_depth(data: np.ndarray, bit_depth: int) -> bytes:
    """
    Pack an integer array into a compact bitstream.

    Each array element is encoded using exactly ``bit_depth`` bits.
    Array elements are packed in row-major (C-order) format.

    Parameters
    ----------
    data : np.ndarray
        Integer input array to pack.

    bit_depth : int
        Number of bits used to represent each element.

    Returns
    -------
    bytes
        Packed binary bitstream.

    Raises
    ------
    ValueError
        If ``bit_depth`` is not positive or if array values
        exceed the representable range.
    """
    if bit_depth <= 0:
        raise ValueError("Bit depth must be positive")

    max_val = (1 << bit_depth) - 1

    if np.any(data < 0) or np.any(data > max_val):
        raise ValueError(
            "Input values exceed representable range "
            f"for bit depth {bit_depth}"
        )

    ba = bitarray(endian='big')

    flat_data = data.flatten(order='C')

    for value in flat_data:
        ba.extend(
            int2ba(
                int(value),
                length=bit_depth,
                endian='big'
            )
        )

    return ba.tobytes()

def unpack_from_bit_depth(byte_stream: bytes, bit_depth: int, shape: tuple[int, ...]) -> np.ndarray:
    """
    Unpack a fixed-width integer bitstream into an array.

    Array elements are reconstructed assuming row-major
    (C-order) packing.

    Parameters
    ----------
    byte_stream : bytes
        Packed binary bitstream.

    bit_depth : int
        Number of bits used to represent each element.

    shape : tuple[int, ...]
        Shape of the reconstructed array.

    Returns
    -------
    np.ndarray
        Reconstructed array with dtype ``uint64``.

    Raises
    ------
    ValueError
        If ``bit_depth`` is not positive.
    """
    if bit_depth <= 0:
        raise ValueError("Bit depth must be positive")

    ba = bitarray(endian='big')
    ba.frombytes(byte_stream)

    total_elements = np.prod(shape)

    unpacked = np.zeros(total_elements, dtype=np.uint64)

    for i in range(total_elements):
        start = i * bit_depth
        end = start + bit_depth

        unpacked[i] = ba2int(ba[start:end])

    return unpacked.reshape(shape, order='C')