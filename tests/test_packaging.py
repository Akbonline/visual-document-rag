from importlib.resources import files


def test_pep561_marker_is_packaged_with_the_library() -> None:
    assert files("vidore_rag").joinpath("py.typed").is_file()
