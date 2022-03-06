"""Multihot dataset."""

import random
import tensorflow as tf
import tensorflow_datasets as tfds
import json
import numpy as np
import pathlib
import re
import cv2
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image

DIM = (300,300)
SHAPE = (300,300,3)

# TODO(shot): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Positions
"""

# TODO(shot): BibTeX citation
_CITATION = """
none
"""

class Position(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Multihot dataset."""
  VERSION = tfds.core.Version('1.0.0')

  BUILDER_CONFIGS = [
    tfds.core.BuilderConfig(name='prod'),
    tfds.core.BuilderConfig(name='dev'),
  ]

  def label_for_training(self):
    if self.builder_config == None:
      return 'prod'
    else:
      return self.builder_config.name

  def labels(self):
    return sorted([
      'small-green-apple', 
      'large-red-apple', 
      'small-green-banana', 
      'large-yellow-banana', 
      'small-yellow-grapes', 
      'large-green-grapes'
    ])
  
  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=SHAPE),
            #TODO revise this to multihot
            'label': tfds.features.ClassLabel(names=self.labels()),
        }),
        supervised_keys=('image','label'),  # Set to `None` to disable
        homepage='https://no.no/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = pathlib.Path('.')

    return {
      'train': self._generate_examples(path),
    }

  def _generate_examples(self, path):
    """Yields examples."""

    for lab in self.labels():
      files = list(path.glob("images/"+lab+"/*.jpg"))
      #TODO encode a multihot
      #hot = lab.split('-')
      #attr = {'label':dict()}
      #for h in hot:
      #  attr['label'][h] = True
      for f in files:
        attr = dict()
        attr['image'] = f
        attr['label'] = lab
        yield str(f), attr
