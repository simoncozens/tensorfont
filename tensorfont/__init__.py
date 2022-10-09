
import numpy as np
import math
import freetype
import uharfbuzz as hb

from skimage.transform import resize
from skimage import filters

from tensorfont.interpshape import interp_shape
from scipy import ndimage
import scipy

from skimage.morphology import convex_hull_image
from scipy.ndimage import distance_transform_edt, distance_transform_cdt

from functools import lru_cache
from itertools import tee

from fontTools import ttLib

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

  def __init__(self, filename, x_height_in_px=None):
    self.filename = filename
    self.face = freetype.Face(filename)

    # Set size by rendering /x and dividing by its height
    if x_height_in_px is None:
      self.scale_factor = 1
    else:
      self.face.set_pixel_sizes(0, self.face.units_per_EM) # render at 1 pixel per unit
      x_height_at_em = self.get_xheight()
      self.scale_factor = x_height_in_px / x_height_at_em
    x_height_unit = int(self.face.units_per_EM * self.scale_factor)
    self.face.set_pixel_sizes(0, x_height_unit)

    self.ascender = self.face.ascender
    self.ascender_px = int(self.face.ascender * self.scale_factor)
    """The font's ascender height, in font units and pixels."""

    self.descender = self.face.descender
    self.descender_px = int(self.face.descender * self.scale_factor)
    """The font's descender height, in font units and pixels (usually negative)."""

    self.full_height = self.ascender - self.descender
    self.full_height_px = self.ascender_px - self.descender_px
    """The font's full (descender + ascender) height, in font units and pixels."""

    self.baseline_ratio = 1 - (self.ascender) / self.full_height
    """The ascender-to-descender ratio."""

    self.ttFont = ttLib.TTFont(self.filename)
    self.hbFont = None
    self.glyphcache = {}

  def get_xheight(self):
    self.face.load_char("x", freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_MONO)
    return self.face.glyph.bitmap.rows

  def glyph(self, g):
    """Access a glyph by name. Returns a `Glyph` object."""
    if g in self.glyphcache: return self.glyphcache[g]
    self.glyphcache[g] = Glyph(self,g)
    return self.glyphcache[g]

  @property
  def italic_angle(self):
    """The italic angle of the font, in degrees."""
    return -(self.ttFont["post"].italicAngle)

  @property
  def m_width(self):
    """The width of the 'm' glyph, in font units."""
    return self.glyph("m").width

  @property
  def x_height(self):
    """The height of the 'x' glyph, in font units."""
    return self.glyph("x").height

  def pair_kerning(self, left, right):
    """The kerning between two glyphs (specified by name), in font units."""
    if self.face.has_kerning:
      return (self.face.get_kerning(left, right).x >> 6) * self.scale_factor
    else:
      if not self.hbFont:
        with open(self.filename, "rb") as fontfile:
          fontdata = fontfile.read()
        face = hb.Face(fontdata)
        font = hb.Font(face)
        scale = face.upem * self.scale_factor
        font.scale = (scale, scale)
        self.hbFont = font

      buf = hb.Buffer()
      buf.add_str(left+right)
      buf.guess_segment_properties()
      hb.shape(self.hbFont, buf, {"kern":True})
      pos = buf.glyph_positions[0].x_advance
      buf = hb.Buffer()
      buf.add_str(left+right)
      buf.guess_segment_properties()
      hb.shape(self.hbFont, buf, {"kern":False})
      pos2 = buf.glyph_positions[0].x_advance
      return pos-pos2

  def pair_distance(self, left, right, with_kerning=True):
    """The ink distance between two named glyphs, in font units.
    This is formed by adding the right sidebearing of the left glyph to the left sidebearing
    of the right glyph, plus a kerning correction. To turn off kerning, use `with_kerning=False`."""
    distance = self.glyph(left).rsb + self.glyph(right).lsb
    if with_kerning:
      distance = distance + self.pair_kerning(left, right)
    return distance

  def minimum_ink_distance(self,left,right):
    """The distance, in pixels, between the ink of the left glyph and the ink of the right glyph, when
    sidebearings are discarded. For many pairs, this will be zero, as the shapes bump up against
    each other (consider "nn" and "oo"). However, pairs like "VA" and "xT" will have a large
    minimum ink distance."""
    right_of_l = self.glyph(left).as_matrix().right_contour()
    left_of_r  = self.glyph(right).as_matrix().left_contour()
    return np.min(right_of_l + left_of_r)

  def maximum_ink_distance(self,left,right):
    """The maximum distance, in pixels, between the ink of the left glyph and the ink of the right glyph, when
    sidebearings are discarded. In other words, the size of the "hole" in the glyph (LV has a large hole)."""
    right_of_l = self.glyph(left).as_matrix().right_contour(max_depth=-1)
    left_of_r  = self.glyph(right).as_matrix().left_contour(max_depth=-1)
    return np.max(right_of_l + left_of_r)

  def shift_distances(self,l,r,dist):
    """Returns two distances, for which the left glyph matrix and the right glyph matrix
    should be translated such that, when the translations are done, the pair is set at a
    distance `dist` pixels apart.

    (Inputs `l` and `r` are glyph names, not `GlyphRendering` objects.)
    """
    sample_distance = dist + self.minimum_ink_distance(l,r)
    sample_distance_left = np.ceil(sample_distance / 2)
    sample_distance_right = np.floor(sample_distance / 2)
    total_ink_width = self.glyph(l).ink_width + self.glyph(r).ink_width
    ink_width_left = np.floor(total_ink_width / 4)
    ink_width_right = np.ceil(total_ink_width / 4)
    total_width_at_minimum_ink_distance = total_ink_width - self.minimum_ink_distance(l, r)
    left_translation = (-(np.ceil(total_width_at_minimum_ink_distance/2) + sample_distance_left) - (-ink_width_left))
    right_translation = ((np.floor(total_width_at_minimum_ink_distance/2) + sample_distance_right) - ink_width_right)
    return left_translation,right_translation

  def set_string(self, s, pair_distance_dict = {}):
    """Returns a matrix containing a representation of the given string. If a dictionary
    is passed to `pair_distance_dict`, then each pair name `(l,r)` will be looked up
    in the directionary and the result will be used as a distance *in pixel units* at
    which to set the pair. If no entry is found or no dictionary is passed, then the font
    will be queried for the appropriate distance.

    Hint: If you want to use glyph names which are not single characters, then pass an
    *array* of glyph names instead of a string."""
    def pairwise(iterable):
      a, b = tee(iterable)
      next(b, None)
      return zip(a, b)
    image = self.glyph(s[0]).as_matrix()
    for l,r in pairwise(s):
      newimage = self.glyph(r).as_matrix()
      if (l,r) in pair_distance_dict:
        dist = pair_distance_dict[(l,r)]
      else:
        dist = self.pair_distance(l,r)
      image = image.impose(newimage,int(dist))
    return image


class Glyph(object):
  """A representation of a glyph and its metrics."""
  def __init__(self, font, g):
    self.font = font
    self.face = font.face

    self.name = g
    """The name of the glyph."""

    gid = self.font.ttFont.getGlyphID(g)
    self.face.load_glyph(gid)
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
    self.tsb       = self.font.ascender_px - slot.bitmap_top # could use max(0, ...) here
    """The top sidebearing (distance from ascender to ink top) in font units."""

  @lru_cache(maxsize=1000)
  def as_matrix(self, normalize = False, binarize = False):
    """Renders the glyph as a matrix. By default, the matrix values are integer pixel greyscale values
    in the range 0 to 255, but they can be normalized or turned into binary values with the
    appropriate keyword arguments. The matrix is returned as a `GlyphRendering` object which
    can be further manipulated."""
    box_height = self.font.full_height_px
    gid = self.font.ttFont.getGlyphID(self.name)
    self.face.load_glyph(gid)

    slot = self.face.glyph
    bitmap = slot.bitmap

    top = slot.bitmap_top # above-baseline glyph height
    w, h = bitmap.width, bitmap.rows

    y = max(int(self.tsb), 0) # top-most row to draw on

    visible_height = min(h, box_height - y) - max(-self.tsb, 0)

    Z = np.zeros((box_height, w))
    Z[y:y+visible_height, 0:w] += np.array(bitmap.buffer, dtype='ubyte').reshape(h, w)[max(-self.tsb, 0):visible_height+max(-self.tsb, 0), :].astype(np.float)
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

  def normalize(self):
    self /= self.max()
    return self

  def transform(self, func):
    return GlyphRendering.init_from_numpy(self._glyph, func(self))

  def with_padding(self, left_padding, right_padding):
    """Returns a new `GlyphRendering` object, left and right zero-padding to the glyph image. If the padding is negative, the matrix will be trimmed appropriately."""
    matrix = self
    if left_padding < 0:
        matrix = GlyphRendering.init_from_numpy(self._glyph,self[:,-left_padding:])
        left_padding = 0
    if right_padding < 0:
        matrix = GlyphRendering.init_from_numpy(self._glyph,self[:,:right_padding])
        right_padding = 0
    padding = ((0,0),(left_padding, right_padding))
    return matrix.transform(lambda x: np.pad(x, padding, "constant"))

  def with_padding_to_constant_box_width(self, box_width):
    padding_width = (box_width - self._glyph.ink_width) / 2.0
    return self.with_padding(int(np.ceil(padding_width)), int(np.floor(padding_width)))

  def with_padding_to_size(self,box_height,box_width):
    padding_width = (box_width - self.shape[1]) / 2.0
    padding_height = (box_height - self.shape[0])
    padding_height_top = self._glyph.font.baseline_ratio * padding_height
    padding_height_bottom = (1-self._glyph.font.baseline_ratio) * padding_height
    padding = ((int(np.ceil(padding_height_top)), int(np.floor(padding_height_bottom))), (int(np.ceil(padding_width)), int(np.floor(padding_width))))
    if np.any(np.array(padding) <0):
      print(padding)
      return None
    return self.transform(lambda x: np.pad(x, padding, "constant"))

  def with_center_cropped_to_size(self, box_height, box_width):
    def crop_center(img,cropx, cropy):
        y,x = img.shape
        if x > cropx:
          startx = x//2-(cropx//2)
          img = img[:,startx:startx+cropx]
        if y > cropy:
          starty = y//2-(cropy//2)
          img = img[starty:starty+cropy, :]
        return img
    return self.transform(lambda x: crop_center(x, box_width, box_height))

  def set_to_constant_size(self, box_height, box_width):
    return self.with_center_cropped_to_size(box_height, box_width).with_padding_to_size(box_height, box_width)

  def with_sidebearings(self):
    """Returns a new `GlyphRendering` object, extending the image to add the
    glyph's sidebearings. If the sidebearings are negative, the matrix
    will be trimmed appropriately."""
    lsb = self._glyph.lsb
    rsb = self._glyph.rsb
    return self.with_padding(lsb,rsb)

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
    return GlyphRendering.init_from_numpy(self._glyph, resize(self, (height, new_width), mode="constant"))

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

  def apply_flexible_distance_kernel(self, strength):
    """Transforms the matrix by applying a flexible distance kernel, with given strength."""
    transformed = 1. - np.clip(strength-distance_transform_edt(np.logical_not(self)),0,strength)
    return GlyphRendering.init_from_numpy(self._glyph,transformed)

  def gradients(self):
    """Returns a pair of images representing the horizontal and vertical gradients."""
    return filters.sobel_h(self), filters.sobel_v(self)

  def impose(self, other, distance=0):
    """Returns a new `GlyphRendering` object made up of two `GlyphRendering` objects
    placed side by side at the given distance."""
    if self.shape[0] != other.shape[0]:
      raise ValueError("heights don't match in impose")
    extension = distance + other.shape[1]
    extended = self.with_padding(0,extension)
    maxofboth = min(np.max(self), np.max(other))
    extended[:,self.shape[1]+distance:self.shape[1]+extension] += other
    return extended.transform(lambda x: np.clip(x, 0, maxofboth))

  def set_at_distance(self,other,distance=0):
    """Similar to `impose` but returns a pair of `GlyphRendering` objects separately, padded at the correct distance."""
    s2, o2 = self.with_padding(0, other.shape[1] + distance), other.with_padding(self.shape[1]+distance, 0)
    return s2, o2

  def rgb(self):
    """Return a three-channel representation."""
    return np.repeat(self[:,:,np.newaxis], 3, axis=2)

  def mask_ink_to_edge(self):
    """Returns two `GlyphRendering` objects representing the left and right "edges" of the glyph:
    the first has positive values in the space between the left-hand contour and the left edge of the matrix
    and zero values elsewhere, and the second has positive values between the right-hand contour and
    the right edge of the matrix and zero values elsewhere. In other words this gives you the
    "white" at the edge of the glyph, without any interior counters."""
    def last_nonzero(arr, axis, invalid_val=-1):
        mask = arr > 5/255.0
        val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
        return np.where(mask.any(axis=axis), val, invalid_val)

    def first_zero(arr, axis, invalid_val=-1):
        mask = arr < 5/255.0
        val = mask.argmax(axis=axis) - 1
        return np.where(mask.any(axis=axis), val, invalid_val)

    def left_counter(image):
        lcounter = 1 - image
        lnonz = first_zero(lcounter,1)
        for x in range(lnonz.shape[0]): lcounter[x,1+lnonz[x]:] = 0
        lcounter -= np.min(lcounter)
        for x in range(lnonz.shape[0]): lcounter[x,lnonz[x]:] = 0
        return lcounter

    def right_counter(image):
        rcounter = np.flip(image,axis=1)
        rcounter = left_counter(rcounter)
        rcounter = np.flip(rcounter,axis=1)
        return rcounter

    return [GlyphRendering.init_from_numpy(self._glyph,x) for x in [left_counter(self), right_counter(self)]]

  def discontinuity(self, contour="left", tolerance = 0.05):
    """Provides a measure, from zero to one, of the "jumpiness" or discontinuity of a
    contour. By default it looks at the left contour; pass `contour="right"` to look
    at the right. You can feed this value to `reduce_concavity`. The tolerance parameter
    determines the size of "jumps" that are significance, measured in fractions of an em.
    By default, anything less that 0.05em is considered smooth and continuous."""
    if contour == "left":
        c = self.left_contour(max_depth=-1)
    else:
        c = self.right_contour(max_depth=-1)
    steps = c[1:] - c[:-1]
    non_edges = ((c[:-1]!=-1) & (c[1:]!=-1))
    jumps = np.abs(steps) > (self._glyph.font.m_width * tolerance)
    total = np.abs(np.sum(steps * non_edges * jumps))
    total2 = np.sum(np.abs(steps * non_edges * jumps))

    return np.clip(total / self.shape[1],0,1)

  def right_face(self, epsilon=0.2):
    c = self.right_contour(max_depth=-1)
    c = np.delete(c,np.where(c==-1))
    pctile = np.min(c) + 0.2 * (np.max(c)-np.min(c))
    return np.sum(c < pctile) / c.shape[0]

  def left_face(self, epsilon=0.2):
    c = self.left_contour(max_depth=-1)
    c = np.delete(c,np.where(c==-1))
    pctile = np.min(c) + 0.2 * (np.max(c)-np.min(c))
    return np.sum(c < pctile) / c.shape[0]

  def reduce_concavity(self,percent):
    """Smooths out concavities in glyphs like 'T', 'L', 'E' by interpolating between the
    glyph and its convex hull."""

    ch = GlyphRendering.init_from_numpy(self,convex_hull_image(self))
    r = self.left_contour(max_depth=-1)
    rch = ch.left_contour(cutoff=0.1,max_depth=-1)
    interpolated = (rch * (percent) + r * (1-percent)).astype(np.int32)
    new = GlyphRendering.init_from_numpy(self,self.copy())
    fill = np.max(new)
    for line in range(0,len(interpolated)):
        if interpolated[line] != -1:
            new[line,interpolated[line]:r[line]+1] = fill

    r = self.right_contour(max_depth=-1)
    rch = ch.right_contour(cutoff=0.1,max_depth=-1)
    width = self.shape[1]
    interpolated = (rch * (percent) + r * (1-percent)).astype(np.int32)
    for line in range(0,len(interpolated)):
        if interpolated[line] != -1:
            new[line,(width-r[line]+1):(width-interpolated[line])] = fill

    return GlyphRendering.init_from_numpy(self._glyph,new)
