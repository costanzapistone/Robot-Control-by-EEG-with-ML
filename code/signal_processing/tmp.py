

class Dataset(object):
    def __init__(self, file_path):
        """
        Load the MATLAB file and extract the data.
        """
        # Load the MATLAB file
        x = None

        # Preprocess
        x = self._segment(x)
        x = self._fft(x)
        x = self._lda(x)
        self._data = x

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)

    def _segment(self, x):
        pass

    def _fft(self, x):
        pass

    def _lda(self, x):
        pass
                


class Class(object):
    def __init__(self):
        pass

    def train(self, dataset: Dataset):
        pass

    def evaluate(self, dataset: Dataset):
        pass

    def predict_action(self, x):
        pass
