import numpy as np

from m3cv_prep.array_tools import pack_array_sparsely, unpack_sparse_array


def test_pack_unpack_sparse_array():
    # Create a sample 3D array
    original_array = np.zeros((4, 4, 4), dtype=int)
    original_array[1, 1, 1] = 1
    original_array[2, 2, 2] = 1
    original_array[3, 3, 3] = 1

    # Pack the array sparsely
    rows, cols, slices = pack_array_sparsely(original_array)

    # Unpack the sparse representation back to a dense array
    unpacked_array = unpack_sparse_array(rows, cols, slices, original_array.shape)

    # Assert that the unpacked array matches the original array
    assert np.array_equal(
        original_array, unpacked_array
    ), "Unpacked array does not match the original"
