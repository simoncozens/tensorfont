tensorfont
----------

Tensorfont is a library to help those performing numerical analysis of
font data, particular with reference to letterfitting and spacing.

Here is an example session::

    >>> from tensorfont import Font
    >>> f = Font("../atospace/kern-dump/normal/MrsEavesOT-Bold.otf")

    >>> f.m_width
    828.0
    >>> f.baseline_ratio
    0.28835063437139563

    >>> f.pair_distance("A","V")
    -149
    >>> f.pair_kerning("A","V")
    -73

    >>> f.glyph("G").lsb
    71

    >>> m = f.glyph("G").as_matrix()
        .with_sidebearings()
        .crop_descender()
        .scale_to_height(50)
    >>> m.left_contour()
    array([25, 22, 20, 18, 16, 15, 14, 13, 12, 11, 10, 10,  9,  9,  8,  8,  7,
            7,  7,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,
            7,  7,  8,  8,  9, 10, 10, 11, 12, 13, 13, 15, 16, 17, 19, 21])
    >>> plt.imgshow(m) ; plt.show()
    # The letter "G" is shown

Full documentation is available at https://simoncozens.github.io/tensorfont/index.html
