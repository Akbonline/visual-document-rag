from vidore_rag.ocr.tesseract import _parse_tsv


def test_tesseract_tsv_preserves_words_lines_and_geometry() -> None:
    tsv = "\n".join(
        [
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
            "5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t96.5\tBattery",
            "5\t1\t1\t1\t1\t2\t45\t20\t20\t12\t91.0\thot",
            "5\t1\t1\t1\t2\t1\t10\t40\t25\t12\t89.0\tStop",
        ]
    )

    words, text = _parse_tsv(tsv)

    assert text == "Battery hot\nStop"
    assert len(words) == 3
    assert (words[0].x, words[0].y, words[0].width) == (10, 20, 30)
