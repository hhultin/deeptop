import math

env_seed = 100
min_charge = 1
max_charge = 8
min_deadline = 1
max_deadline = 12
theta = 1
mu = 0.5
sigma = 0.5
dt = 1
x0 = 0.5

state_dim = 2
action_dim = 1
num_iter = int(1e6)
num_runs = 5
log_interval = 10000
