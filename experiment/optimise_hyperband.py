import numpy as np

from random import random
from math import log, ceil
from time import time, ctime


from hyperopt import hp
from hyperopt.pyll.stochastic import sample

from train_wrapper import run_trial
# sampling function signature: qloguniform(loglow, loghigh, rounding factor)
# loguniform(loglow, loghigh)
# loglow and loghigh written like this- log(x) to show what's the actual range of x
space = {
    'lr': hp.loguniform( 'lr', log(1e-4), log(1e-1)), 
    'gamma': 1-hp.loguniform( 'discount', log(1e-5), log(1)),
    'batch_size': hp.qloguniform( 'batch_size', log(32), log(2048), 16),
    'num_units': hp.qloguniform( 'num_units', log(32), log(128), 16 )
}

def get_params():
    #"learning rate-lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    #"discount factor- gamma", type=float, default=0.95, help="discount factor")
    #"batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    #"num-units", type=int, default=64, help="number of units in the mlp")

	params = sample( space )
	# round floats, need to do this because qloguniform has rounding errors with small numbers
	params['lr']=round(params['lr'], 5)
	params['gamma']=round(params['gamma'], 5)
	# enforce integers
	params['batch_size']=ceil(params['batch_size'])
	params['num_units']=ceil(params['num_units'])
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
						reward = run_trial(scenario, n_iterations*self.n_episode_per_iter, t)
						result = {'loss':-reward, 'log_loss':0}
						
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
		
		return self.results

hb = Hyperband(get_params, max_iter=1000)
results = hb.run('simple_tag')