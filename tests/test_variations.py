from tensorfont import Font


def test_variation_mwidth():
    font = Font("tests/data/Roboto[wdth,wght].ttf")
    at_default = font.m_width
    assert at_default == 1796

    font = Font("tests/data/Roboto[wdth,wght].ttf", variations={"wght": 100})
    at_wght_100 = font.m_width
    assert at_wght_100 == 1836

    font = Font("tests/data/Roboto[wdth,wght].ttf", variations={"wdth": 75})
    at_wdth_75 = font.m_width
    assert at_wdth_75 == 1542


def test_variation_kerning():
    font = Font("tests/data/Roboto[wdth,wght].ttf")
    at_default = font.pair_kerning("A", "V")

    font = Font(
        "tests/data/Roboto[wdth,wght].ttf", variations={"wdth": 75, "wght": 800}
    )
    at_wdth_75 = font.pair_kerning("A", "V")
    assert at_wdth_75 != at_default
