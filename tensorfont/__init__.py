
import numpy as np
import math
import freetype

from skimage.transform import resize
from skimage.util import pad

from tensorfont.getKerningPairsFromOTF import OTFKernReader
from functools import lru_cache

safe_glyphs = set([
  "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
  "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
   ])

safe_glyphs_l = set([
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
])  

safe_glyphs_r = set([
  "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
  "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
])


class Font(object):
  """The `Font` module is your entry point for using Tensorfont.
  Use this to load up a font and begin exploring it.
  """

  def __init__(self, filename):
    self.face = freetype.Face(filename)
    self.face.set_char_size( 64 * self.face.units_per_EM )

    self.ascender = self.face.ascender
    """The font's ascender height, in font units."""

    self.descender = self.face.descender
    """The font's descender height, in font units (usually negative)."""

    self.full_height = self.ascender - self.descender
    """The font's full (descender + ascender) height."""

    self.baseline_ratio = 1 - (self.ascender) / self.full_height
    """The ascender-to-descender ratio."""

    self.glyphcache = {}
    self.kernreader = OTFKernReader(filename)

  def glyph(self,g):
    """Access a glyph by name. Returns a `Glyph` object."""
    if g in self.glyphcache: return self.glyphcache[g]
    self.glyphcache[g] = Glyph(self,g)
    return self.glyphcache[g]

  @property
  def m_width(self):
    """The width of the 'm' glyph, in font units."""
    return self.glyph("m").width

  @property
  def x_height(self):
    """The height of the 'x' glyph, in font units."""
    return self.glyph("x").height

  def pair_kerning(self,left,right):
    """The kerning between two glyphs (specified by name), in font units."""
    if self.face.has_kerning:
      return self.face.get_kerning(left, right).x >> 6
    else:
      if (left,right) in self.kernreader.kerningPairs:
        return self.kernreader.kerningPairs[(left,right)]
      return 0

  def pair_distance(self,left,right,with_kerning=True):
    """The ink distance between two named glyphs, in font units.
    This is formed by adding the right sidebearing of the left glyph to the left sidebearing
    of the right glyph, plus a kerning correction. To turn off kerning, use `with_kerning=False`."""
    distance = self.glyph(left).rsb + self.glyph(right).lsb
    if with_kerning:
      distance = distance + self.pair_kerning(left,right)
    return distance

  def minimum_ink_distance(self,left,right):
    """The distance between the ink of the left glyph and the ink of the right glyph, when
    sidebearings are discarded. For many pairs, this will be zero, as the shapes bump up against
    each other (consider "nn" and "oo"). However, pairs like "VA" and "xT" will have a large
    minimum ink distance."""
    right_of_l = self.glyph(left).as_matrix().right_contour()
    left_of_r  = self.glyph(right).as_matrix().left_contour()
    return np.min(right_of_l + left_of_r)

class Glyph(object):
  """A representation of a glyph and its metrics."""
  def __init__(self, font, g):
    self.font = font
    self.face = font.face

    self.name = g
    """The name of the glyph."""

    n = self.face.get_name_index(g.encode("utf8"))
    self.face.load_glyph(n, freetype.FT_LOAD_RENDER |
                              freetype.FT_LOAD_TARGET_MONO)
    slot = self.face.glyph
    self.ink_width = slot.bitmap.width
    self.ink_height= slot.bitmap.rows
    self.width     = slot.metrics.horiAdvance / 64
    """The width of the glyph in font units (including sidebearings)."""
    self.height    = slot.metrics.height / 64.0
    """The height of the glyph in font units."""
    self.lsb       = slot.bitmap_left
    """The left sidebearing in font units."""
    self.rsb       = ((slot.advance.x >> 6) - slot.bitmap.width - self.lsb)
    """The right sidebearing in font units."""
    self.tsb       = max(self.font.ascender - slot.bitmap_top,0)
    """The top sidebearing (distance from ascender to ink top) in font units."""

  @lru_cache(maxsize=1000)
  def as_matrix(self, normalize = False, binarize = False):
    """Renders the glyph as a matrix. By default, the matrix values are integer pixel greyscale values
    in the range 0 to 255, but they can be normalized or turned into binary values with the
    appropriate keyword arguments. The matrix is returned as a `GlyphRendering` object which
    can be further manipulated."""
    box_height = self.font.full_height
    self.face.set_char_size( 64 * self.face.units_per_EM )
    self.face.load_char(self.name)

    slot = self.face.glyph
    bitmap = slot.bitmap

    top = slot.bitmap_top # above-baseline glyph height
    w, h = bitmap.width, bitmap.rows

    y = int(self.tsb) # top-most row to draw on

    visible_height = min(h, box_height - y)

    Z = np.zeros((box_height, w))
    Z[y:y+visible_height, 0:w] += np.array(bitmap.buffer, dtype='ubyte').reshape(h,w)[0:visible_height, :].astype(np.float)
    if normalize or binarize:
      Z = Z / 255.0
    if binarize:
      Z = Z.astype(int)
    return GlyphRendering.init_from_numpy(self,Z)

class GlyphRendering(np.ndarray):
  @classmethod
  def init_from_numpy(self, glyph, matrix):
    s = GlyphRendering(matrix.shape)
    s[:] = matrix
    s._glyph = glyph
    return s

  def with_padding(self, left_padding, right_padding):
    """Returns a new `GlyphRendering` object, left and right zero-padding to the glyph image."""
    padding = ((0,0),(left_padding,right_padding))
    padded = pad(self, padding, "constant")
    return GlyphRendering.init_from_numpy(self._glyph, padded)

  def with_sidebearings(self):
    """Returns a new `GlyphRendering` object, extending the image to add the
    glyph's sidebearings. If the sidebearings are negative, the matrix
    will be trimmed appropriately."""
    lsb = self._glyph.lsb
    rsb = self._glyph.rsb
    matrix = self
    if lsb < 0:
        matrix = GlyphRendering.init_from_numpy(self._glyph,self[:,-lsb:])
        lsb = 0
    if rsb < 0:
        matrix = GlyphRendering.init_from_numpy(self._glyph,self[:,:rsb])
        rsb = 0
    return matrix.with_padding(lsb,rsb)

  def mask_to_x_height(self):
    """Returns a new `GlyphRendering` object, cropping the glyph image
    from the baseline to the x-height. (Assuming that the input `GlyphRendering` is full height.)"""
    f = self._glyph.font
    baseline = int(f.full_height + f.descender)
    top = int(baseline - f.x_height)
    cropped = self[top:baseline,:]
    return GlyphRendering.init_from_numpy(self._glyph, cropped)

  def crop_descender(self):
    """Returns a new `GlyphRendering` object, cropping the glyph image
    from the baseline to the ascender. (Assuming that the input `GlyphRendering` is full height.)"""
    f = self._glyph.font
    baseline = int(f.full_height + f.descender)
    cropped = self[:baseline,:]
    return GlyphRendering.init_from_numpy(self._glyph, cropped)

  def scale_to_height(self, height):
    """Returns a new `GlyphRendering` object, scaling the glyph image to the
    given height. (The new width is calculated proportionally.)"""
    new_width = int(self.shape[1] * height / self.shape[0])
    return GlyphRendering.init_from_numpy(self._glyph, resize(self, (height, new_width)))

  def left_contour(self, cutoff = 30, max_depth = 10000):
    """Returns the left contour of the matrix; ie, the 'sidebearing array' from the
    edge of the matrix to the leftmost ink pixel. If no ink is found at a given
    scanline, the value of max_depth is used instead."""
    contour = np.argmax(self > cutoff, axis=1) + max_depth * (np.max(self, axis=1) <= cutoff)
    return np.array(contour)

  def right_contour(self, cutoff = 30, max_depth = 10000):
    """Returns the right contour of the matrix; ie, the 'sidebearing array' from the
    edge of the matrix to the rightmost ink pixel. If no ink is found at a given
    scanline, the value of max_depth is used instead."""
    pixels = np.fliplr(self)
    contour = np.argmax(pixels > cutoff, axis=1) + max_depth * (np.max(pixels, axis=1) <= cutoff)
    return np.array(contour)
