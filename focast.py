# focast
#
# T. Warren de Wit
# University of Alabama Huntsville
# warrendewit.com
#
# Performs a shape-matching-based algorithm for predicting
# stock prices and displays the resulting performance.

#----------------------------------------------------
# Setup
#----------------------------------------------------

import logging
log = "focast.log"

# Clear out the old log.
with open(log, 'w'):
    pass

# Initialize the log.
logging.basicConfig(filename=log,level=logging.INFO)

feature_reserve = {}
result_reserve = {}

#----------------------------------------------------
# Focast Methods
#----------------------------------------------------

def get_feature(
	stock = "LINC",
	monday = "2018-01-1",
	window_count = 1,
	max_days = 15,
	window_size = 8,
	correlation_threshold = 0.8,
	pips_to_match = 6,
	shapebook = [
		{
			"name": "Head-and-Shoulder",
			"id": 1,
			"vector": [6.5,2.5,4.5,1,4.5,2.5,6.5]
		},
		{
			"name": "Triple-Top",
			"id": 2,
			"vector": [6.5,2,4.5,2,4.5,2,6.5]
		},
		{
			"name": "Double-Top",
			"id": 3,
			"vector": [6.5,4.5,1.5,3,1.5,4.5,6.5]
		},
		{
			"name": "Spike-Top",
			"id": 4,
			"vector": [4.5,4.5,4.5,1,4.5,4.5,4.5]
		},
		{
			"name": "Spike-Top-Reversed",
			"id": 5,
			"vector": [3.5,3.5,3.5,7,3.5,3.5,3.5]
		},
		{
			"name": "Double-Top-Reversed",
			"id": 6,
			"vector": [1.5,3.5,6.5,5,6.5,3.5,1.5]
		},
		{
			"name": "Triple-Top-Reversed",
			"id": 7,
			"vector": [1.5,6,3.5,6,3.5,6,1.5]
		},
		{
			"name": "Head-and-Shoulder-Reversed",
			"id": 8,
			"vector": [1.5,5.5,3.5,7,3.5,5.5,1.5]
		}
	]):

	if stock in feature_reserve:
		if monday in feature_reserve[stock]:
			return feature_reserve[stock][monday]
	else:
		feature_reserve[stock] = {}

	'''
	Gets a feature vector for teh given stock on the given monday.
	'''

	#----------------------------------------------------
	# Initialization
	#----------------------------------------------------

	# Figure out the monday we're getting the feature for
	# as a datetime so we can operate on it.

	from dateutil import parser
	monday_date = parser.parse(monday)

	# Get a logger
	logger = logging.getLogger("get_feature")

	#----------------------------------------------------
	# Get the daily prices for the previous trading days.
	#----------------------------------------------------

	# Getting twice the data length means that we'll have
	# for sure the requested number of trading days.
	#
	# Remember that we're looking for all days BEFORE this one,
	# up to max_days.

	import datetime

	query_start = (monday_date - datetime.timedelta(days = 2*max_days+1)).strftime("%Y-%m-%d")
	query_end = (monday_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

	# We want the daily price data.

	from yahoofinancials import YahooFinancials
	query = YahooFinancials([stock])
	query_results = query.get_historical_price_data(query_start,query_end,"daily")cd
	prices = query_results[stock]["prices"]

	# We only want max_days number of prices.
	prices = prices[-max_days:]

	#----------------------------------------------------
	# Look for the shapes in the price data.
	#----------------------------------------------------

	# Start a list of matches.
	matches = []

	# Look through the data in window_size increments.
	from scipy.stats import pearsonr, rankdata

	for i in range(len(prices)-window_size):

		# Get the current window of trading day prices.
		window_prices = prices[i:i+window_size]

		# Get the closing prices for each day.
		window_prices_close = [price["close"] for price in window_prices]

		# Convert the prices to an index of points.
		price_points = [(j, window_prices_close[j]) for j in range(len(window_prices_close))]

		# Get the PIP points for the current price points.
		import fastpip
		pip_points = fastpip.pip(price_points,7)

		# Extract just the PIP values.
		pips = [pip_point[1] for pip_point in pip_points]

		# Get the rank vector for the PIPs.
		pip_rank_vector = [float(i) for i in rankdata(pips)]

		# Try to find the best matching shape (if any) against the PIPs.
		best_match = None

		# Try each shape to test its correlation.
		for shape in shapebook:

			# Get the correlation to this shape.
			correlation = pearsonr(shape["vector"][:pips_to_match],pip_rank_vector[:pips_to_match])[0]

			# Only consider this shape if it is above the correlation threshold.
			if correlation > correlation_threshold:

				# The start and end dates for the match are determined
				# by the start and end dates of the window.
				match_start = window_prices[0]["formatted_date"]
				match_end = window_prices[-1]["formatted_date"]

				# The "days ago" is how many days before the monday prediction date
				# that the shape began on.
				#
				# This should be at least 3 (for the weekend).
				match_start_date = parser.parse(match_start)
				match_days_ago = (monday_date - match_start_date).days

				# Store all the match information
				match = {
					"shape_name": shape["name"],
					"shape_id": shape["id"],
					"correlation": correlation,
					"start_date": match_start,
					"end_date": match_end,
					"days_ago": match_days_ago
					}

				# If there is already a match, see if we're the best.
				# If not, we're the best by default,
				#
				# since we've established that we're above the
				# correlation threshold.

				if not best_match:
					best_match = match
				else:
					if match["correlation"] > best_match["correlation"]:
						best_match = match

		# If there was a sufficient match, add it.
		if best_match:
			matches.append(best_match)

	#----------------------------------------------------
	# Get the most recent shapes.
	#----------------------------------------------------

	# Trim to the most recent matches.
	if len(matches) > window_count:
		matches = matches[-window_count:]

	#----------------------------------------------------
	# Turn the shapes into a vector.
	#----------------------------------------------------

	vector = []

	# Pad with zeroes for the missing matches.
	null_match_count = window_count - len(matches)
	for i in range(null_match_count):
		vector += [0.0, 0.0, 0.0]

	# Add the match data.
	for match in matches:
		vector += [match["shape_id"], match["days_ago"], match["correlation"]]

	feature_reserve[stock][monday] = vector

	# The full vector is now assembled.
	return vector

def get_result(
	stock = "LINC",
	monday = "2018-01-1",
	wisdom_threshold = 1.025):
	
	'''
	Gets the result of a given week of trading for a given stock.
	Returns a dictionary with:
		- start_price
		- end_price
		- wisdom
		- profitable
		- change
	'''

	if stock in result_reserve:
		if monday in result_reserve[stock]:
			return result_reserve[stock][monday]
	else:
		result_reserve[stock] = {}

	#----------------------------------------------------
	# Initialization
	#----------------------------------------------------

	# Figure out the monday we're getting the result for
	# as a datetime so we can operate on it.

	from dateutil import parser
	monday_date = parser.parse(monday)

	# Get a logger
	logger = logging.getLogger("get_result")

	#----------------------------------------------------
	# Get the week's worth of trading history.
	#----------------------------------------------------

	# We just want the week's worth of trading information

	import datetime

	query_start = monday_date.strftime("%Y-%m-%d")
	query_end = (monday_date + datetime.timedelta(days=5)).strftime("%Y-%m-%d")

	from yahoofinancials import YahooFinancials
	query = YahooFinancials([stock])

	query_results = query.get_historical_price_data(query_start,query_end,"daily")

	prices = query_results[stock]["prices"]

	#----------------------------------------------------
	# Determine the results and labels.
	#----------------------------------------------------

	# We're using the Monday(Open)/Friday(Close) price structure.
	# Indexing using start and end so that we can account for any holidays, etc.
	start_price = prices[0]["open"]
	end_price = prices[-1]["close"]

	# Get the change.
	change = end_price / start_price

	# Determine if it's profitable.
	profitable = 0
	if change > 1:
		profitable = 1

	# Determine if it's "wise".
	wisdom = 0
	if change > wisdom_threshold:
		wisdom = 1

	# Get a dictionary with the results.

	result = {
		"start_price": start_price,
		"end_price": end_price,
		"wisdom": wisdom,
		"profitable": profitable,
		"change": change
		}

	result_reserve[stock][monday] = result

	return result

def get_predictions(
	stocks = ["LINC"],
	monday = "2018-01-1",
	training_weeks = 12):

	'''
	Makes predictions for the given stocks on the given monday.

	Returns a dictionary keyed on stocks,
	where each value is a 1 (Wise) or 0 (Unwise)
	'''

	#----------------------------------------------------
	# Initialization
	#----------------------------------------------------

	# Figure out the monday we're getting the prediction for
	# as a datetime so we can operate on it.

	from dateutil import parser
	monday_date = parser.parse(monday)

	# Get a logger
	logger = logging.getLogger("get_predictions")

	#----------------------------------------------------
	# Get training mondays
	#----------------------------------------------------

	import datetime

	# Get a list of mondays to train on.
	training_mondays = []

	# Go back that number of weeks,
	# since we assume we're being passed in a monday.
	for i in range(training_weeks):
		weeks_ago = i+1
		training_monday_date = monday_date - datetime.timedelta(weeks=weeks_ago)
		training_monday = training_monday_date.strftime("%Y-%m-%d")
		training_mondays.append(training_monday)

	#----------------------------------------------------
	# Get training data
	#----------------------------------------------------

	# Get the features and label vectors.
	features = []
	labels = []

	# Get the training data,
	# for each stock,
	# for each week.

	for stock in stocks:
		for training_monday in training_mondays:

			feature = get_feature(stock,training_monday)
			label = get_result(stock,training_monday)["wisdom"]

			features.append(feature)
			labels.append(label)

			logger.info("Predicting "+stock+" "+monday+" - Training "+stock+" "+training_monday + " = "+str(feature)+" : "+str(label))

	
	#----------------------------------------------------
	# Make the classifier
	#----------------------------------------------------

	from sklearn.ensemble import RandomForestClassifier
	from xgboost import XGBClassifier
	import numpy as np

	classifier = RandomForestClassifier()
	classifier.fit(features,labels)

	# classifier = XGBClassifier()
	# classifier.fit(np.array(features),np.array(labels))

	#----------------------------------------------------
	# Run the predictions
	#----------------------------------------------------

	# Get the predictions dictionary.
	predictions = {}

	# Get the prediction for each stock.
	for stock in stocks:
		feature = get_feature(stock,monday)

		label = 0
		if 0.0 not in feature:
			label = classifier.predict([feature])[0]
			
		predictions[stock] = label

	return predictions

#----------------------------------------------------
# Experiments
#----------------------------------------------------

def run_2018(wealth_list):

	'''
	Runs a 35-selected stock list over all mondays in 2018.
	Prints the wealth achieved as it goes.
	'''

	print("Monday,deltaWeek,deltaTotal,deltaUniform,WiseTP,WiseFP,WiseFN,WiseTN,ProfitTP,ProfitFP,ProfitFN,ProfitTN")

	import sys

	# Get a logger
	logger = logging.getLogger("run_2018")

	# List of stocks for the experiment.
	# Using the ones I've been using for most of the project.
	stock_list = [
			"LINC",
			"RGLS",
			"OTIC",
			"KMPH",
			"LPTX",
			"HOS",
			"RESN",
			"CVM",
			"OSG",
			"PESI",
			"NBEV",
			"APTO",
			"GSUM",
			"ANY",
			"AEZS",
			"CASI",
			"PRPL",
			"ASFI",
			"ATOS",
			"INPX",
			"MHLD",
			"IPDN",
			"HDSN",
			"MLNT",
			"NLNK",
			"QHC",
			"MYOS",
			"OXBR",
			"GEN",
			"MCF",
			"RSLS",
			"AAC",
			"BNTC",
			"AKER",
			"ESEA"
		]

	# Get all mondays in 2018.

	mondays = []

	import datetime

	date = datetime.date(2018,1,1)
	date += datetime.timedelta(days=1-date.isoweekday())

	while date.year == 2018:
		mondays.append(date.strftime("%Y-%m-%d"))
		date += datetime.timedelta(days=7)

	# mondays = mondays[22:42]

	# Keep track of uniformed prices
	starts = {}

	# Run the experiment.

	wealth = 1.0


	for monday in mondays:

		# Get predictions for this monday.
		predictions = get_predictions(stock_list,monday)

		# Get the results for each stock.
		# Keep track of which ones we're investing in.

		wise_tp = 0
		wise_fp = 0
		wise_fn = 0
		wise_tn = 0

		profit_tp = 0
		profit_fp = 0
		profit_fn = 0
		profit_tn = 0

		uniform_change = 0.0

		invested_results = []
		for stock in stock_list:
			result = get_result(stock,monday)

			if monday == mondays[0]:
				starts[stock] = result["start_price"]

			logger.info("Predicted: "+str(predictions[stock])+" | Actual: "+str(result))

			if predictions[stock]:
				invested_results.append(result)

			# Wisdom prediction accuracy

			if predictions[stock] and result["wisdom"]:
				wise_tp += 1

			if predictions[stock] and not result["wisdom"]:
				wise_fp += 1

			if not predictions[stock] and result["wisdom"]:
				wise_fn += 1

			if not predictions[stock] and not result["wisdom"]:
				wise_tn += 1

			# Profit prediction accuracy

			if predictions[stock] and result["profitable"]:
				profit_tp += 1

			if predictions[stock] and not result["profitable"]:
				profit_fp += 1

			if not predictions[stock] and result["profitable"]:
				profit_fn += 1

			if not predictions[stock] and not result["profitable"]:
				profit_tn += 1

			# Track uniform investment
			uniform_change += (result["end_price"] / starts[stock]) * (1.0 / float(len(stock_list)))

		weekly_change = 1.0

		# Do the wealth management.
		if invested_results:
			weekly_change = 0.0
			for result in invested_results:
				weekly_change +=  result["change"] / float(len(invested_results))
			wealth *= weekly_change

		data = []
		data.append(monday)
		data.append(weekly_change)
		data.append(wealth)
		data.append(uniform_change)
		data.append(wise_tp)
		data.append(wise_fp)
		data.append(wise_fn)
		data.append(wise_tn)
		data.append(profit_tp)
		data.append(profit_fp)
		data.append(profit_fn)
		data.append(profit_tn)

		line = ','.join([str(d) for d in data])

		print line
		sys.stdout.flush()

	wealth_list.append(wealth)


#----------------------------------------------------
# Main Program
#----------------------------------------------------

total_wealths = []
for i in range(500):
	run_2018(total_wealths)

print ""
print "Wealth Summary"

for w in total_wealths:
	print w

print ""
print "Average Wealth Achieved"
print float(sum(total_wealths))/len(total_wealths)

import numpy as np

print ""
print "Stdev Wealth Achieved"
print np.std(total_wealths)