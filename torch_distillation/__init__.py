from .data import (GLUE_TASKS, GLUE_TASKS_MAPPING, GLUETaskDataset,
                   GroupedBatchSampler, LanguageModelingDataset)
from .distiller import Distiller, HintonDistiller, SanhDistiller
from .loss import HintonLoss, SanhLoss, SoftTargetLoss

__version__ = '0.0.1'
__all__ = [
    'Distiller',
    'GLUE_TASKS',
    'GLUE_TASKS_MAPPING',
    'GLUETaskDataset',
    'GroupedBatchSampler',
    'HintonDistiller',
    'HintonLoss',
    'LanguageModelingDataset',
    'SanhDistiller',
    'SanhLoss',
    'SoftTargetLoss'
]
