import dists
import csv

class Searcher:
	def __init__(self, dbPath):
		# store the database path (path to index.csv file)
		self.dbPath = dbPath

	# queryFeature : extracted features from the query image
	# numResults : max nubmer of results to return
	def search(self, queryFeatures, numResults = 10):
		# initialize the results dictionary
		results = {}

		# open the database for reading
		with open(self.dbPath) as f:
			reader = csv.reader(f)

			# loop over the rows in the index
			for row in reader:
				# parse out the image ID and features, then compute the
				# chi-squared distance between the features in our database
				# and the query features
				features = [float(x) for x in row[1:]]
				d = dists.chi2_distance(features, queryFeatures)

				# now that we have the distance between the 2 feature vectors,
				# we can update the results dictionary
				#	key: current imageID in the database
				#	value: distance we just compute, representing how 'similar'
				#		   the image in the database is to our query
				results[row[0]] = d

			# close the reader
			f.close()
	
		# sort the results, smaller distances (more relevant images)
		# are at the front of the list
		results = sorted([(v, k) for (k, v) in results.items()])

		# return the results
		return results[:numResults]

