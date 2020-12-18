from tensorfont.dataset import get_test_font
import tensorfont
import random


class RandomPair:
    def __init__(
        self,
        box_height,
        box_width,
        x_height,
        fonts,
        left_glyphs=None,
        right_glyphs=None,
        perturbation=[0, 1],
        kerned=True,
    ):
        self.box_height = box_height
        self.box_width = box_width
        self.perturbation = perturbation
        self.kerned = kerned

        if isinstance(fonts, str):
            assert fonts == "training" or fonts == "validation"
            self.get_font = lambda: get_test_font(fonts, x_height)
        else:
            self.get_font = lambda: tensorfont.dataset._cached_font(
                random.choice(fonts), x_height
            )

        if left_glyphs is not None:
            self.left_glyphs = left_glyphs
        else:
            self.left_glyphs = list(tensorfont.safe_glyphs_l)
        if right_glyphs is not None:
            self.right_glyphs = right_glyphs
        else:
            self.right_glyphs = list(tensorfont.safe_glyphs_r)

    def generator(self):
        while True:
            f = self.get_font()
            l = random.choice(self.left_glyphs)
            r = random.choice(self.right_glyphs)
            # Check they exist in font
            glyphs = f.ttFont.getGlyphOrder()
            if l in glyphs and r in glyphs:
                yield (f, l, r)

    def get_image(self, f, l, r, perturbation_range=[0, 1], kerning=True):
        perturbation = random.randrange(*perturbation_range) * f.scale_factor
        pdd = {}
        pdd[(l, r)] = f.pair_distance(l, r, with_kerning=kerning) - perturbation
        return (
            f.set_string([l, r], pdd)
            .normalize()
            .set_to_constant_size(self.box_height, self.box_width),
            perturbation,
        )

    def image_generator(self):
        gen = self.generator()
        while True:
            f, l, r = next(gen)
            img, perturbation = self.get_image(f, l, r, self.perturbation, self.kerned)
            if img.shape == (self.box_height, self.box_width):
                yield (img, perturbation)
