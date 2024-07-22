import math

env_seed = 100
cap = 1000
order_size = 500
demand_list = []
for t in range(10):
    demand_list.append(math.sin(math.pi * t / 10.0) * 300)
selling_price = 20

state_dim = 1
action_dim = 1
num_iter = int(1e6)
num_runs = 5
log_interval = 10000
