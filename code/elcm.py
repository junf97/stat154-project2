class ELCM():
	def __init__(self, sd_threshold=2.0, corr_threshold=0.75, ndai_threshold=0.215):
		self.sd_threshold = sd_threshold
		self.corr_threshold = corr_threshold
		self.ndai_threshold = ndai_threshold

	def fit(self, X, Y):
		# we can tune the threshold NDAI here.
		pass

	def predict(self, X):
		return ((X["SD"]<self.sd_threshold) | (X["CORR"]>self.corr_threshold) & (X["NDAI"]<self.ndai_threshold)).apply(lambda bool: -1 if bool else 1)
		


