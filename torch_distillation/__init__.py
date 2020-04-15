from .data import GroupedBatchSampler, LanguageModelingDataset, chunk, quantize
from .distiller import Distiller, HintonDistiller, SanhDistiller
from .loss import HintonLoss, SanhLoss, SoftTargetLoss

__version__ = '0.0.1'
__all__ = [
    'chunk',
    'Distiller',
    'GroupedBatchSampler',
    'HintonDistiller'
    'HintonLoss',
    'LanguageModelingDataset',
    'quantize',
    'SanhDistiller',
    'SanhLoss',
    'SoftTargetLoss'
]
