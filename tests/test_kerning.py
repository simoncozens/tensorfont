from tensorfont import Font


def test_kerning_names():
    font = Font("tests/data/Roboto[wdth,wght].ttf")
    av = font.pair_kerning("A", "V")
    aacute_v = font.pair_kerning("Aacute", "V")
    assert av != 0
    assert av == aacute_v
