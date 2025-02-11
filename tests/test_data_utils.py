"""Unit tests for I3Extractor class."""

from graphnet.data.utils import is_i3_file, has_extension


def test_is_i3_file():
    """Test `is_i3_file_` function.
    """
    assert is_i3_file("path/to/file.i3.bz2") is True
    assert is_i3_file("path/to/file.bz2") is True
    assert is_i3_file("path/to/file.zst") is True
    assert is_i3_file("path/to/GCD_file.i3.gz") is False
    assert is_i3_file("path/to/gcd_file.i3.zst") is False
    assert is_i3_file("path/to/GEO_file.i3.zst") is False
    assert is_i3_file("path/to/geo_file.i3.gz") is False

def test_has_extension():
    extensions = ["i3.bz2","zst","gz"]
    assert has_extension("path/to/file.i3.bz2", extensions) is True
    assert has_extension("path/to/file.bz2", extensions) is False
    assert has_extension("path/to/file.i3.gz", extensions) is True
    assert has_extension("path/to/file.gz", extensions) is True
    assert has_extension("path/to/file.zst", extensions) is True
    assert has_extension("path/to/file.zst.txt", extensions) is False
    assert has_extension("path/to/file_gz.csv", extensions) is False
