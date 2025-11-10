
class Dataset():
    """
    An abstract class for the data. Provisional    
    """
    def __init__(self):
        pass

    @property
    def data(self):
        return self._data

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def __len__(self):
        return len(self._data)

    def __getitem__(self):
        pass

    def array_to_dataframe(self):
        pass

    def drop_na(self):
        pass

    def fill_na(self):
        pass

    def normalize(self):
        pass

    def denormalize(self):
        pass