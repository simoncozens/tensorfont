"""
Download training font dataset from Google fonts
------------------------------------------------

Google Fonts makes it easy for us to download a large number of freely
licensed fonts, suitable for use as a dataset. This module downloads and
caches a number of fonts, then splits them into a training and validation
set. The fonts are stored in the ``.tensorfont`` directory in the user's
home directory.
"""

import os
import requests
import zipfile
import glob
import random
import functools
from tensorfont import Font

__all__ = ["prepare_training_data", "get_test_font"]

full_families = [
    "Roboto",
    "Open Sans",
    "Lato",
    "Montserrat",
    "Oswald",
    "Raleway",
    "Poppins",
    "Ubuntu",
    "Merriweather",
    "PT Sans",
    "Playfair Display",
    "Work Sans",
    "Lora",
    "Nanum Gothic",
    "Titillium Web",
    "Anton",
    "Barlow",
    "Teko",
    "Crimson Text",
    "Libre Baskerville",
    "Source Serif Pro",
    "Arvo",
    "Merriweather Sans",
    "IBM Plex Sans",
    "EB Garamond",
    "Zilla Slab",
    "IBM Plex Serif",
    "Martel",
    "Fira Sans Condensed",
]


def font_dir(suffix=""):
    if suffix: suffix = suffix + "/"
    fontdir = os.path.expanduser("~/.tensorfont/" + suffix)
    if not os.path.isdir(fontdir):
        os.mkdir(fontdir)
    return fontdir


def download_zip(url, name):
    zip_path = os.path.join(font_dir(), name + ".zip")
    try:
        if not os.path.isfile(zip_path):
            with open(zip_path, "wb") as f:
                print(f"Downloading {name}")
                response = requests.get(url, stream=True)
                total_length = response.headers.get("content-length")

                if total_length is None:  # no content length header
                    f.write(response.content)
                else:
                    dl = 0
                    total_length = int(total_length)
                    for data in response.iter_content(chunk_size=4096):
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        sys.stdout.write("\r[%s%s]" % ("=" * done, " " * (50 - done)))
                        sys.stdout.flush()

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(font_dir())
    except KeyboardInterrupt as e:
        print("Unlinking")
        if os.path.isfile(zip_path):
            os.unlink(zip_path)
        raise e


def download_families(families):
    for f in families:
        download_zip("https://fonts.google.com/download?family=" + f, f)


def splitfiles(validation_split=0.3):
    font_dir()
    train = font_dir("training")
    validation = font_dir("validation")
    for f in glob.glob(font_dir() + "/*.?tf"):
        if random.random() <= validation_split:
            dest = validation
        else:
            dest = train
        os.rename(f, os.path.join(dest, os.path.basename(f)))


def prepare_training_data(validation_split=0.3, families=None):
    """Downloads fonts and splits them into training and validation sets

    Args:
        validation_split (float): Proportion of files to set aside for validation.
        families: List of font families to download. If not provided a default, hand-picked set will be downloaded.
    """
    if families is None:
        families = full_families
    download_families(families = families)
    splitfiles(validation_split=validation_split)


@functools.lru_cache(maxsize=100)
def _cached_font(filename, x_height_in_px=None):
    return Font(filename, x_height_in_px)

def get_test_font(source="training", x_height_in_px = None):
    """Returns a random font from dataset as a ``tensorfont.Font`` object

    Args:
        source: "training" or "validation"
        x_height_in_px: Font x-height in pixels, or none for upem scaling.
    """
    file = random.choice( glob.glob(font_dir(source) + "*.?tf") )
    return _cached_font(file, x_height_in_px)
