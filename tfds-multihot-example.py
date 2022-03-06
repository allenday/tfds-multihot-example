"""Multihot dataset."""

import numpy as np
import pathlib
import tensorflow_datasets as tfds

DIM = (300,300)
SHAPE = (300,300,3)

# TODO(shot): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Multihot
"""

# TODO(shot): BibTeX citation
_CITATION = """
none
"""

class Multihot(tfds.core.GeneratorBasedBuilder):
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
        'small','large',
        'red','yellow','green',
        'apple','banana','grapes'
    ])
  
  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=SHAPE),
            'label': tfds.features.FeatureConnector.from_json(
                    {'type': 'tensorflow_datasets.core.features.tensor_feature.Tensor', 'content': {'shape': [len(self.labels())], 'dtype': 'bool'} })
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

    ii = 0
    mh = dict()

    dd = ""
    for lab in self.labels():
      for token in lab.split('-'):
        dd += token + " "
        if token not in mh:
          mh[token] = ii
          ii += 1
    #vectorizer = TextVectorization(output_mode="binary")
    #vectorizer.adapt(np.array([[dd]]))
    #integer_data = vectorizer([[dd]])
    #print(integer_data)

    for f in list(path.glob("images/*/*.jpg")):
      parts = str(f).split('/')
      lab = parts[1]
      mhot = np.zeros(len(self.labels()), dtype=np.bool)
      for h in lab.split('-'):
        mhot[mh[h]] = True
      yield str(f), {'label':mhot,'image':f}
