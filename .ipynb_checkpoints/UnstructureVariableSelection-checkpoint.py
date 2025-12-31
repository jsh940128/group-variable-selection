from .LabelBVS import LabelBVS
from .WordFrequencyGenerator import WordFrequencyGenerator

class UnstructureVariableSelection(LabelBVS, WordFrequencyGenerator):
    def __init__(self, 
                 data,
                 review_col,
                 rating_col,
                 aggregate_col,
                 word_corpus_file,
                 dist_type = 'cooccur', 
                 track = False):
        WordFrequencyGenerator.__init__(
            self, data, review_col, rating_col, aggregate_col, word_corpus_file
        )
        self.dist_type = dist_type
        self.track = track

    def generate(self, word_count = None, top_n = None):
        X, y = super().generate(word_count, top_n)
        LabelBVS.__init__(
            self, 
            X = X,
            y = y['mean_rating'],
            weight = y['count'],
            dist_type = self.dist_type,
            track = self.track
        )

        return X, y