import abc


class InvalidDataDirectory(Exception):
    """
    Error raised when the chosen intput directory for the dataset is not valid.
    """


class BaseReader(object):
    """Base reader for reading different types of data
    """
    def __init__(self, **kwargs):
        super(BaseReader, self).__init__()

    @property
    @abc.abstractproperty
    def total(self):
        """Returns the total amount of records in the dataset.
        """

    @abc.abstractmethod
    def iterate(self):
        """Iterates over the records in the dataset.
        """
