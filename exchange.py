from krakenex import *
import time
from environment import *
from agent import *
from itertools import chain
from time import gmtime, strftime
import csv 


api_key = 'gnVoWu8DBhJ38yZgsm7wvM+VA4osPYKK5zZs9WnwL02sVgKCNxaTNjZK'
p_key = 'wMyvEYCmQQzatENszqUweV/ibzfOk8APk0FY/D6wqLE5qVq4nesp4nMiiwG7OfWP/dtlXXpX8luYyPVCZHOjcw=='

def main():
	#collect_obs_to_file("ETHUSD,07_22", 'ETHUSD', 3.1)
	samplingBackup = {
		'name': 'sampling'
	}
	backup = {
		'name': 'sampling'
	}
	env = Environment('ETHUSD,07_07.csv', setup=False, time=True)
	rl_agent(env, range(0,201,10)[1:], backup, 2, 500, 20, 5)
	measure_profitability_path(env, [200], samplingBackup, 2, 200, 10, 10)


# def generate_states

def collect_obs_to_file(file_name, ticker, frequency):
	print ('Collecting orderbooks') 
	ob_file = open(file_name + '.csv', 'w')
	writer = csv.writer(ob_file)
	connection = API(key=api_key, secret=p_key)
	req = {
		'pair': ticker,
		'count': 20		
	}
	while True:
		try:
			result = connection.query_public('Depth', req)
			print (strftime("%Y-%m-%d %H:%M:%S", gmtime()))
			asks =  result['result']['XETHZUSD']['asks']
			bids =  result['result']['XETHZUSD']['bids']
			write_csv(writer, asks, bids)
		except:
			print('Call Failed')
			pass
		time.sleep(frequency)


def write_csv(writer, asks, bids):
	asks = [x for level in asks for x in level[0:2]]
	bids = [x for level in bids for x in level[0:2]]
	interleaved = asks + bids
	if len(interleaved) % 4 != 0:
		print (len(interleaved))
	else:
		interleaved[::4] = asks[::2]
		interleaved[1::4] = asks[1::2]
		interleaved[2::4] = bids[::2]
		interleaved[3::4] = bids[1::2]
		interleaved.append(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
		writer.writerow(interleaved)

if __name__ == "__main__":
	main()