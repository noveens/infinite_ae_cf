hyper_params = {
	'dataset': 'ml-1m', 
	'float64': False,

	'depth': 1,
	'grid_search_lamda': True,
	'lamda': 1.0, # Only used if grid_search_lamda == False

	# Number of users to keep (randomly)
	'user_support': -1, # -1 implies use all users
	'seed': 42,
}
