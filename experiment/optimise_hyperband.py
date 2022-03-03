import numpy as np

from random import random
from math import log, ceil
from time import time, ctime
import pickle, os


from hyperopt import hp
from hyperopt.pyll.stochastic import sample

from train_wrapper import run_trial
# sampling function signature: qloguniform(loglow, loghigh, rounding factor)
# loguniform(loglow, loghigh)
# loglow and loghigh written like this- log(x) to show what's the actual range of x
space = {
    'lr': hp.choice( 'lr', [1e-4, 1e-3, 1e-2]), 
    'gamma': hp.choice('gamma', [0.9, 0.95, 0.99]),
	'batch_size': hp.choice('batch_size', [32, 64]),
    # 'batch_size': hp.choice('batch_size', [32, 64, 128, 256, 512, 1024]),
    'num_units': hp.choice( 'num_units', [32,64,128])
}

def get_params():
    #"learning rate-lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    #"discount factor- gamma", type=float, default=0.95, help="discount factor")
    #"batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    #"num-units", type=int, default=64, help="number of units in the mlp")

	params = sample( space )
	# # round floats, need to do this because qloguniform has rounding errors with small numbers
	# params['lr']=round(params['lr'], 4)
	# params['gamma']=round(params['gamma'], 5)
	# # enforce integers
	# params['batch_size']=ceil(params['batch_size'])
	# params['num_units']=ceil(params['num_units'])
	return params


class Hyperband:
	
	def __init__( self, get_params_function, max_iter ):
		self.get_params = get_params_function
		
		self.max_iter = max_iter  	# maximum iterations per configuration
		self.eta = 3			# defines configuration downsampling rate (default = 3)
		self.n_episode_per_iter = 100

		self.logeta = lambda x: log( x ) / log( self.eta )
		self.s_max = int( self.logeta( self.max_iter ))
		self.B = ( self.s_max + 1 ) * self.max_iter

		self.results = []	# list of dicts
		self.counter = 0
		self.best_loss = np.inf
		self.best_counter = -1
		

	# can be called multiple times
	def run( self, scenario, skip_last = 0, dry_run = False ):
		
		for s in reversed( range( self.s_max + 1 )):
			
			# initial number of configurations
			n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))	
			
			# initial number of iterations per config
			r = self.max_iter * self.eta ** ( -s )		

			# n random configurations
			T = [ self.get_params() for i in range( n )] 
			
			for i in range(( s + 1 ) - int( skip_last )):	# changed from s + 1
				
				# Run each of the n configs for <iterations> 
				# and keep best (n_configs / eta) configurations
				
				n_configs = ceil(n * self.eta ** ( -i ))
				n_iterations = int(r * self.eta ** ( i ))
				
				print("\n*** {} configurations x {:.1f} iterations of {} episodes each".format( 
					n_configs, n_iterations, self.n_episode_per_iter ))
				
				val_losses = []
				early_stops = []
				
				for t in T:
					
					self.counter += 1
					print("\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format( 
						self.counter, ctime(), self.best_loss, self.best_counter ))
					
					start_time = time()
					
					if dry_run:
						result = { 'loss': random(), 'log_loss': random(), 'auc': random()}
					else:
						reward = run_trial(scenario, n_iterations, self.n_episode_per_iter, t)
						result = {'loss':-reward, 'log_loss':log(-reward)}
						
					assert( type( result ) == dict )
					assert( 'loss' in result )
					
					seconds = int( round( time() - start_time ))
					print("\n{} seconds.".format( seconds ))
					
					loss = result['loss']	
					val_losses.append( loss )
					
					early_stop = result.get( 'early_stop', False )
					early_stops.append( early_stop )
					
					# keeping track of the best result so far (for display only)
					# could do it be checking results each time, but hey
					if loss < self.best_loss:
						self.best_loss = loss
						self.best_counter = self.counter
					
					result['counter'] = self.counter
					result['seconds'] = seconds
					result['params'] = t
					result['iterations'] = n_iterations
					
					self.results.append( result )
				
				# select a number of best configurations for the next loop
				# filter out early stops, if any
				indices = np.argsort( val_losses )
				T = [ T[i] for i in indices if not early_stops[i]]
				T = T[ 0:int( n_configs / self.eta )]
		
		return self.results, T

if __name__ == '__main__':
	scenario = 'simple_speaker_listener'
	hb = Hyperband(get_params, max_iter=81)
	results, T = hb.run(scenario)
	print(results)
	print(T)
	# save stats for each combination of params run by hyperband
	with open('./{}_hyperband_results.pkl'.format(scenario), 'wb') as f:
		pickle.dump(results, f)

	# save error curves
	x_list = []
	for filename in os.listdir('./learning_curves/'):
		with open('./learning_curves/'+filename, 'rb') as f:
			x=pickle.load(f)
			x_list.append(x)
	print(x_list)
	with open('./{}_hyperband_loss_curves.pkl'.format(scenario), 'wb') as f:
		pickle.dump(x_list, f)