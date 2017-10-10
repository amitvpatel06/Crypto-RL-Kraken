from environment import *
from collections import defaultdict
from Q import *
from Qfunction_approx import *
import csv # for reading
import sys
import numpy as np
import math
import multiprocess # for multithreading


class Buffer:

	def __init__(self, look_back, rolling_window=150):
		self.history = []
		self.rolling_window = rolling_window
		self.summands = []
		self.max = -999999999999
		self.min = 999999999999
		self.sum = 0 
		self.look_back = look_back

	def update_rolling_sum(self, next_value):
		self.summands.append(next_value)
		self.sum += next_value
		if len(self.summands) > self.rolling_window :
			self.sum -= self.summands[0]
			self.summands.pop(0)

	def buffer_add(self, addition):
		self.curr = addition
		self.history.append(addition)
		if len(self.history) > self.look_back:
			self.history.pop(0)

		self.max = addition if max(self.history) < addition else max(self.history)
		self.min = addition if min(self.history) > addition else min(self.history)

	def get_zone(self, target, divs):
		if target < self.min:
			return 0
		elif target > self.max:
			return divs
		else:
			return ((target - self.min) // ((self.max - self.min)/(divs - 1)) + 1)

def generate_state(market, divs):
	state = {}
	state['T'] = market['T']
	state['I'] = market['I']
	state['Spread'] = market['Spreads'].get_zone(market['Spreads'].curr, divs)
	state['Misbalance'] = market['Misbalances'].get_zone(market['Misbalances'].curr, divs)
	state['RollingVol'] = market['RollingVol'].get_zone(market['RollingVol'].curr, divs)
	state['RollingSignedVol'] = market['RollingSignedVol'].get_zone(market['RollingSignedVol'].curr, divs)
	return state


def update_market(market):
	diff, net_vol, total_vol = market['book'].ob_diff(market['nextbook'])
	market['currstate'].apply_diff(diff)
	market['Spreads'].buffer_add(market['currstate'].get_spread())
	market['Misbalances'].buffer_add(market['currstate'].get_misbalance())
	market['RollingSignedVol'].update_rolling_sum(net_vol)
	market['RollingSignedVol'].buffer_add(market['RollingSignedVol'].sum)
	market['RollingVol'].update_rolling_sum(total_vol)
	market['RollingVol'].buffer_add(market['RollingVol'].sum)
	market['book'] = market['nextbook']


def generate_data(env, market, start, end):
	market['currstate'] = env.get_book(start)
	market['book'] = env.get_book(start)
	for idx in range(start + 1, end + 1):
		market['nextbook'] = env.get_book(idx)
		update_market(market)

def reset_market(look_back, rolling_window): 
	market = { 
		'I': 0,
		'Spreads': Buffer(look_back),
		'Misbalances': Buffer(look_back),
		'RollingVol': Buffer(look_back),
		'RollingSignedVol': Buffer(look_back)
	}
	return market



def measure_profitability_path(env, times, backup, units, look_back, rolling_window, divs):
	table = Q(1, backup)
	percent_table = Q(1, backup)
	# state variables
	time = {}
	profits = {}
	T = times[0]
	market = reset_market(look_back, rolling_window)
	market['currstate'] = env.get_book(0)
	market['book'] = env.get_book(0)
	S = 1000
	# train algorithm to buy
	for T in times:
		print (T)
		time['T'] = T
		for ts in range(look_back, look_back + S):
			if ts > look_back:
				market = reset_market(look_back, rolling_window)
				market['T'] = T
				generate_data(env, market, ts - look_back, ts)
				cost = env.vol_order(market['currstate'], 0, units)
				s = generate_state(market, divs)
				for t in range(T):
					market['nextbook'] = env.get_book(ts + t + 1)
					update_market(market)
				exit = market['currstate'].immediate_cost_sell(units)
				neg_percent_profit = -1 * (exit - cost)/cost * 100
				table.update_table(s, 1, neg_percent_profit)
				if neg_percent_profit < -0.5:
					percent_table.update_table(s, 1, 0)
				else:
					percent_table.update_table(s, 1, 1)
					
	# train algorithm to sell
	for T in times:
		print (T)
		time['T'] = T 
		for ts in range(look_back, look_back + S):
			if ts > look_back:
				market = reset_market(look_back, rolling_window)
				market['T'] = T
				generate_data(env, market, ts - look_back, ts)
				short = env.vol_order(market['currstate'], 1, units)
				s = generate_state(market, divs)
				exited = False
				for t in range(T):
					market['nextbook'] = env.get_book(ts + t + 1)
					update_market(market)
					close = market['currstate'].immediate_cost_buy(units)
				neg_percent_profit = -1 * (short - close)/short * 100
				table.update_table(s, -1, neg_percent_profit)
				if neg_percent_profit < -0.5:
					percent_table.update_table(s, -1, 0)
				else:
					percent_table.update_table(s, -1, 1)

	current_time = {}
	profit_table = Q(1, backup)
	for T in times:
		market['T'] = T
		current_time['T'] = T
		for ts in range(look_back + S, look_back + S + 5000):
			if ts % 100 == 0:
				print(ts)
			if ts > look_back:
				market = reset_market(look_back, rolling_window)
				market['T'] = T
				generate_data(env, market, ts - look_back, ts)
				s = generate_state(market, divs)
				action, value = table.arg_min(s)
				if value <  0.2:
					# choose position to take based on the action suggested by table
					exited = False
					side = 0 if action == 1 else 1
					enter = env.vol_order(market['currstate'], side, units)
					for t in range(T):
						market['nextbook'] = env.get_book(ts + t + 1)
						update_market(market)
						# determine profitability based on side
						if side == 0:
							# long position exit
							exit = market['currstate'].immediate_cost_sell(units)
							percent = (exit - enter)/enter * 100
						else: 
							# short position exit
							exit = market['currstate'].immediate_cost_buy(units)
							percent = (enter - exit)/enter * 100
						if percent > 0.52:
							profit_table.update_table(current_time, 1, percent)
							exited = True 
							break
					if not exited:
						profit_table.update_table(current_time, 1, percent)
	import pdb
	pdb.set_trace()


def rl_agent(env, times, backup, units, look_back, rolling_window, divs):
	table = Q(1, backup)
	percent_table = Q(1, backup)
	# state variables
	time = {}
	profits = {}
	T = max(times)
	market = reset_market(look_back, rolling_window)
	market['currstate'] = env.get_book(0)
	market['book'] = env.get_book(0)
	S = 1000
	print (list(times))
	backup_transitions = []
	order_books = len(env.books)
	random.seed(1)

	for side in range(2):
		opposite_side = (side + 1) % 2
		action = pow(-1, side)
		for ex in range(look_back, look_back + S):
			ts = random.randint(look_back + 2, order_books - (T + 10))
			if ex % 100 == 0:
				backup_trs(table, backup_transitions)
				backup_transitions = []
				print(ts)
			# set up the simulation and collect the lookback data
			market = reset_market(look_back, rolling_window)
			generate_data(env, market, ts - look_back, ts)
			# calculate initial position cost and value
			initial_p = env.vol_order(market['currstate'], side, units)
			market['T'] = 0
			s_0 = generate_state(market, divs)
			market['I'] = pow(-1, side) 
			state_number = 0
			
			for t in range(T + 1):
				market['nextbook'] = env.get_book(ts + t + 1)
				update_market(market)
				if t in times:
					market['T'] = t
					state_number += 1
					s_curr = generate_state(market, divs)
					if state_number == 1:
						curr_pos_v = market['currstate'].immediate_value(opposite_side, units)
						neg_percent_profit = pow(-1, side + 1) * (curr_pos_v - initial_p)/initial_p * 100
						backup_transitions.append([s_0, action, neg_percent_profit, curr_pos_v, initial_p])
					else: 
						next_pos_v = market['currstate'].immediate_value(opposite_side, units)
						neg_percent_change = pow(-1, side + 1) * (next_pos_v - curr_pos_v)/curr_pos_v * 100
						backup_transitions.append([s_prev, 1, neg_percent_change, next_pos_v, curr_pos_v])
						curr_pos_v = next_pos_v
					s_prev = s_curr
		backup_trs(table, backup_transitions)
		 

	current_time = {}
	profit_table = Q(1, backup)
	pred = []
	act = []
	
	for ex in range(0, 2000):
		ts = random.randint(look_back, order_books - (T + 10))
		if ex % 100 == 0:
			print(ts)
		if ts >= look_back:
			market = reset_market(look_back, rolling_window)
			market['T'] = 0
			generate_data(env, market, ts - look_back, ts)
			s_0 = generate_state(market, divs)
			action, value = table.arg_min(s_0)
			if value < -1:
				pred.append(value)
 				# choose position to take based on the action suggested by table
				side = 0 if action == 1 else 1
				opposite_side = (side + 1) % 2
				market['I'] = pow(-1, side)
				enter_v = env.vol_order(market['currstate'], side, units)
				exited = False
				for t in range(T):
					market['nextbook'] = env.get_book(ts + t + 1)
					update_market(market)
					if t in times:
						market['T'] = t
						curr_s = generate_state(market, divs)
						a, v = table.arg_min(curr_s)
						if v >= 0:
							print ([a, v, t])
							current_time['T'] = T
							exit_v = market['currstate'].immediate_value(opposite_side, units)
							percent = pow(-1, side) * (exit_v - enter_v )/enter_v * 100
							act.append(percent)
							profit_table.update_table(current_time, action, percent)
							exited = True
							break
				if not exited:
					current_time['T'] = T
					exit_v = market['currstate'].immediate_value(opposite_side, units)
					percent = pow(-1, side) * (exit_v - enter_v)/enter_v * 100
					act.append(percent)
					profit_table.update_table(current_time, action, percent)
	import pdb
	pdb.set_trace()



def backup_trs(table, backup_transitions):
	states_done = 0
	for transition in backup_transitions[::-1]:
		state, action, reward, _, _ = transition
		table.update_table(state, 0, 0)
		if states_done == 0:
			#print([state, action, reward, reward])
			table.update_table(state, action, reward)
			next_state_reward = reward
		else:
			backup = reward + min(next_state_reward, 0)
			#print([state, action, reward, backup])
			table.update_table(state, action, backup)
			_, next_state_reward = table.arg_min(state)
		states_done += 1

	 	
def process_output(table, func, executions, T, L):
	"""
	Process output for each run and write to file
	"""
	if table.backup['name'] == 'sampling' or table.backup['name'] == 'replay buffer':
		table_to_write = table.Q
	elif table.backup['name'] == 'doubleQ':
		if func is None:
			table_to_write = table.curr_Q
		else:
			table_to_write = []
			table_to_write.append(table.Q_1)
			table_to_write.append(table.Q_2)
	else:
		print ('agent.dp_algo - invalid backup method')

	tradesOutputFilename = ''
	if not func is None:
		# linear approx model
		tradesOutputFilename += 'linear-'

	tradesOutputFilename += table.backup['name']
	write_trades(executions, tradesOutputFilename=tradesOutputFilename)

	if func is None:
		write_table_files(table_to_write, T, L, tableOutputFilename=table.backup['name'])
	else:
		write_function(table_to_write, T, L, 'linear model',functionFilename=table.backup['name'])

def create_variable_divs(divs, env):
	spreads = []
	misbalances = []
	imm_costs = []
	signed_vols	= []
	if divs > 1:
		spread_diff = (env.max_spread - env.min_spread) * 1.0 / (divs)
		misbalance_diff = (env.max_misbalance - env.min_misbalance) * 1.0 / (divs)
		imm_cost_diff = (env.max_imm_cost - env.min_imm_cost) * 1.0 / (divs)
		signed_vols_diff = (env.max_signed_vol - env.min_signed_vol) * 1.0 / (divs)
		for i in range(1, divs):
			spreads.append(env.min_spread + i * spread_diff)
			misbalances.append(env.min_misbalance + i * misbalance_diff)
			imm_costs.append(env.min_imm_cost + i * imm_cost_diff)
			signed_vols.append(env.min_signed_vol + i * signed_vols_diff)
	spreads.sort()
	misbalances.sort()
	imm_costs.sort()
	signed_vols.sort()
	return spreads, misbalances, imm_costs, signed_vols

def compute_signed_vol(vol, signed_vols):
	if len(signed_vols) == 0 or vol < signed_vols[0]:
		return 0
	for i in range(len(signed_vols) - 1):
		if vol >= signed_vols[i] and vol < signed_vols[i+1]:
			return (i + 1)
	return len(signed_vols)

def compute_imm_cost(curr_book, inv, im_costs):
	if inv == 0:
		return 0
	im_cost = curr_book.immediate_cost_buy(inv)
	if len(im_costs) == 0 or im_cost < im_costs[0]:
		return 0
	for i in range(len(im_costs) - 1):
		if im_cost >= im_costs[i] and im_cost < im_costs[i+1]:
			return (i + 1)
	return len(im_costs)

def compute_bid_ask_spread(curr_book, spreads):
	spread = min(curr_book.a.keys()) - max(curr_book.b.keys())
	if len(spreads) == 0 or spread < spreads[0]:
		return 0
	for i in range(len(spreads) - 1):
		if spread >= spreads[i] and spread < spreads[i+1]:
			return (i + 1)
	return len(spreads)

def compute_volume_misbalance(curr_book, misbalances, env):
	m = env.misbalance(curr_book)
	if len(misbalances) == 0 or m < misbalances[0]:
		return 0
	for i in range(len(misbalances) - 1):
		if m >= misbalances[i] and  m < misbalances[i+1]:
			return (i + 1)
	return len(misbalances)



def write_trades(executions, tradesOutputFilename="run"):
	trade_file = open(tradesOutputFilename + '-trades.csv', 'w')
	# write trades executed
	w = csv.writer(trade_file)
	executions.insert(0, ['Time Left', 'Rounded Units Left', 'Bid Ask Spread', 'Volume Misbalance', 'Immediate Cost', 'Signed Transcation Volume' ,'Action', 'Reward', 'Volume'])
	w.writerows(executions)

def write_function(function, T, L, model,  functionFilename='run'):
	table_file = open(functionFilename + '-' + model + '.csv', 'w')
	tw = csv.writer(table_file)
	table_rows = []
	table_rows.append(['Time Left', 'Rounded Units Left', 'Bid Ask Spread', 'Volume Misbalance', 'Immediate Cost', 'Signed Transcation Volume','Action'])
	if type(function) is list:
		table_rows.append(function[0].coef_)
		table_rows.append(function[1].coef_)
		table_rows.append(function[0].intercept_)
		table_rows.append(function[1].intercept_)
	else:
		table_rows.append(function.coef_)
		table_rows.append(function.intercept_)
	tw.writerows(table_rows)

def write_table_files(table, T, L, tableOutputFilename="run"):
	table_file = open(tableOutputFilename + '-tables.csv', 'w')
	# write table
	tw = csv.writer(table_file)
	table_rows = []
	table_rows.append(['Time Left', 'Rounded Units Left', 'Bid Ask Spread', 'Volume Misbalance', 'Immediate Cost', 'Signed Transcation Volume', 'Action', 'Expected Payout'])
	for key in table:
		for action, payoff in table[key].items():
			if type(action) != str:
				t_left, rounded_unit, spread, volume_misbalance, im_cost, signed_vol = key.split(",")
				table_rows.append([t_left, rounded_unit, spread, volume_misbalance,  im_cost, signed_vol, action, payoff])
	tw.writerows(table_rows)


"""
We here run three backup methods based on how dp tables are updated:
- sampling (simple update)
- double q learning
- replay buffer
"""
if __name__ == "__main__":
	# define method params
	doubleQbackup = {
		'name': 'doubleQ'
	}
	samplingBackup = {
		'name': 'sampling'
	}
	replayBufferBackup = { 'name': 'replay buffer',
					'buff_size': 50,
					'replays': 5
	}
	# tables
	doubleQProcess = multiprocess.Process(target=dp_algo, args=("../data/10_GOOG.csv", 1000, 1000, 10, 10, 10, doubleQbackup, 100000))
	samplingProcess = multiprocess.Process(target=dp_algo, args=("../data/10_GOOG.csv", 1000, 1000, 10, 10, 10, samplingBackup, 100000))
	replayBufferProcess = multiprocess.Process(target=dp_algo, args=("../data/10_GOOG.csv", 1000, 1000, 10, 10, 10, replayBufferBackup, 100000))
	# start
	#doubleQProcess.start()
	samplingProcess.start()
	#replayBufferProcess.start()

	# function approx
	#func_doubleQProcess = multiprocess.Process(target=dp_algo, args=("../data/10_GOOG.csv", 1000, 1000, 10, 10, 10, doubleQbackup, "linear", 100000))
	#func_samplingProcess = multiprocess.Process(target=dp_algo, args=("../data/10_GOOG.csv", 1000, 1000, 10, 10, 10, samplingBackup, "linear", 100000))
	#func_replayBufferProcess = multiprocess.Process(target=dp_algo, args=("../data/10_GOOG.csv", 1000, 1000, 10, 10, 10, replayBufferBackup, "linear", 100000))
	# start
	#func_doubleQProcess.start()
	#func_samplingProcess.start()
	#func_replayBufferProcess.start()
