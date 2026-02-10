class QualityCalculator(object):
    def __init__(self):
        pass
    
    def fit(self, frequency_matrix, beta):
        self.frequency_matrix = frequency_matrix
        self.beta = beta
        return self
    
    def transform(self, frequency_matrix):
        return frequency_matrix @ self.beta
    
    def fit_transform(self, frequency_matrix, beta):
        self.fit(frequency_matrix, beta)
        return self.transform(frequency_matrix)