import numpy as np

# compute the chi-squared distance, which is used to compare 
# discrete probability distributions
# since we are comparing color histograms, which are by definition
# probability distributions, chi-squred function is an excellent choices
def chi2_distance(histA, histB, eps = 1e-10):
	d = 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))
	return d
