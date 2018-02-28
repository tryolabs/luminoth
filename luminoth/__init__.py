__version__ = '0.0.4dev0'

__title__ = 'Luminoth'
__description__ = 'Computer vision toolkit based on TensorFlow'
__uri__ = 'https://luminoth.ai'
__doc__ = __description__ + ' <' + __uri__ + '>'

__author__ = 'Tryolabs'
__email__ = 'luminoth@tryolabs.com'

__license__ = 'BSD 3-Clause License'
__copyright__ = 'Copyright (c) 2018 Tryolabs S.A.'

# Import functions that are part of Luminoth's public interface.
from luminoth.cli import cli  # noqa
from luminoth.utils.predicting import PredictorNetwork  # noqa
