import gym, random, time
import tensorflow as tf
import numpy as np


# General Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 1001  # Episode limitation 
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy #change back to 100

HyperParameters
GAMMA =  0.95 # discount factor (Between 0 and 1)
INITIAL_EPSILON =  1.0# starting value of epsilon (Discover Probability)
FINAL_EPSILON =  0. # final value of epsilon
EPSILON_DECAY_STEPS =  23# initial epsilon decreased by 1/x each episode (200 steps)

# Additional HyperParameters: Memory and Training Batch Size
max_memory_size = 5000
max_batch_size = 500
initial_batch_size = 100
max_recent_memory = 1
learning_rate = 0.001

## Parameters reset between runs
step_list = []
memory = []
ave_reward_list = []
total_reward = 0
average_reward = 0
consecutive_wins = 0
stable_flag = 0

##  Parameters related to debugging and runs
max_runs = 1 			# Change to 1 during submission
show_debug = 0 			# Change to 0 during submission
stability_requirement = 100
run_scoreboard = []
failed_scores = []

# Create environment
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# Define Network Graph
hidden_size = 100 
stddev_in = 1.0
stddev_in2 = 0.01 

layer_1_weights = tf.layers.Dense(hidden_size,activation=tf.nn.tanh,
					kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=stddev_in),
					bias_initializer=tf.random_normal_initializer(mean=0.0,stddev=stddev_in)
					)
layer_1 = layer_1_weights.apply(state_in)

output_1_weights = tf.layers.Dense(ACTION_DIM,
					activation=None,
					use_bias=True,
					kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=stddev_in2)
					)


# Network outputs
q_values = output_1_weights.apply(layer_1)

q_action = tf.reduce_sum(tf.multiply(q_values, action_in), axis=1)

# Loss/Optimizer Definition
loss = tf.reduce_mean(tf.square(q_action - target_in))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


def explore(state, epsilon):

    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action 			


#Implements memory storage of previous experiences
def storememory(state, action, reward, next_state, done):
	if len(memory) >= max_memory_size:
		memory.pop(0)
	memory.append((state, action, reward, next_state, done))


#Trains network using recent memory with the size (batch_size)
def trainwithmemory(batch_size, recent_memory, recent_memory_steps):
	
	remaining_space = batch_size - recent_memory_steps
	batch = []

	try: 
		if remaining_space > 1:
			batch = random.sample(memory, remaining_space)
		batch = batch + recent_memory
		for state, action, reward, next_state, done in batch:
		    nextstate_q_values = q_values.eval(feed_dict={state_in: [next_state]})
		    target = reward
		    if not done:
		    	target = reward + GAMMA * np.amax(nextstate_q_values)

		    session.run([optimizer], feed_dict={
	      		target_in: [target],
	            action_in: [action],
	            state_in: [state]
	        })

	except ValueError:
		batch_size -= 1
		trainwithmemory(batch_size, recent_memory, recent_memory_steps)

#Calculates average reward over previous 5 runs
def calc_avgreward(reward):
    	
	if len(ave_reward_list) >= 5:
		ave_reward_list.pop(0)
	ave_reward_list.append(reward)
	return sum(ave_reward_list) // 5

#Returns a list of the most recent memory from the latest episode
def save_recent_memory(recent_memory_steps):
	if len(step_list) >= max_recent_memory:
		step_list.pop(0)
	step_list.append(recent_memory_steps)
	total_steps = sum(a for a in step_list)
	recent_memory = memory[-total_steps:]
	return recent_memory, total_steps

#Changes batchsize depending on the performance of the reward
def dynamicbatchsize(total_reward, train_batch_size, memorysize, episode):
	
	if episode <= 25:
		train_batch_size = memorysize
	elif total_reward < 150:
		train_batch_size += max_batch_size // 4
	elif total_reward < 175:
		train_batch_size = train_batch_size
	elif total_reward <= 200:
		train_batch_size = train_batch_size // 3 

	if train_batch_size >= max_batch_size:
		return max_batch_size
	elif train_batch_size <= 10:
		return 0
	
	return train_batch_size

#Used for debugging purposes. To view code performance over multiple runs
def display_scoreboard(total_runtime):
	win1 = 0
	win2 = 0
	win3 = 0
	avgstability = 0
	avgtime = 0
	print('__________________________________________________________')
	print('Final Scoreboard:')
	print('')
	for x, k in enumerate(run_scoreboard):							
		print('Run {}: {}'.format(x+1,k))
		if k[0]:
			win1 += 1
		if k[1]:
			win2 += 1
		if k[2]:
			win3 += 1
		avgstability += k[3]
		avgtime += k[4]

	perwin1 = round(win1 / max_runs *100,1)
	perwin2 = round(win2 / max_runs *100,1)
	perwin3 = round(win3 / max_runs *100,1)
	avgstability = round(avgstability / max_runs,1)
	avgtime = round(avgtime / max_runs,2)
	print('__________________________________________________________')
	print('Total Runs                   : {}'.format(max_runs))
	print('Total Episodes per Run       : {}'.format(EPISODE-1))	
	print('Total Time                   : {} mins'.format(total_runtime))
	print('Avg Time per Run             : {} mins'.format(avgtime))
	print('>190 Reward on 100th Episode : {}  ({}%)'.format(win1, perwin1))
	print('>190 Reward on 200th Episode : {}  ({}%)'.format(win2, perwin2))
	print('Stable Performance Achieved  : {}  ({}%)'.format(win3, perwin3))
	print('Avg Stable Episode           : {}'.format(avgstability))
	print('__________________________________________________________')
	print('Failed Episode 100 Runs:')
	for y in failed_scores:
		print('Run {}: {}'.format(y[0]+1, y[1]))

# Main learning loop

start_maintime = time.time()

for x in range(max_runs): 
	if show_debug:
		print("Run {} starting:".format(x+1))

	#reset variables
	stable_flag = 0
	stablity_episode = EPISODE
	memory = []
	step_list = []
	ave_reward_list = []
	average_reward = 0
	consecutive_wins = 0
	train_batch_size = initial_batch_size
	epsilon = INITIAL_EPSILON
	start_runtime = time.time()
	run_scoreboard.append([0,0,0,0,0]) 

	# Start session - Tensorflow housekeeping
	session = tf.InteractiveSession()
	session.run(tf.global_variables_initializer())

	for episode in range(EPISODE):
		#initialize task
		state = env.reset()

		# Update epsilon dynamically depending on average reward performance
		if episode > 100 and average_reward < 140:
			if epsilon >= 0.1:
				epsilon = 0.1
			else:
				epsilon += (epsilon / EPSILON_DECAY_STEPS) * 3
		elif epsilon > FINAL_EPSILON:
			epsilon -= epsilon / EPSILON_DECAY_STEPS
		total_reward = 0
		nbstep = 0

		for step in range(STEP):

			action = explore(state, epsilon)
			next_state, reward, done, _ = env.step(np.argmax(action))
			total_reward += reward
			#store step into memory
			storememory(state, action, reward, next_state, done) 
			state = next_state
			nbstep += 1

			if done:
				if total_reward < 199:
					storememory(state, action, reward, next_state, done)
				if total_reward < 180 and consecutive_wins < 35:
					consecutive_wins = 0
				else:
					consecutive_wins += 1
				#Save the most recent experience to ensure that is trained upon
				recent_memory, recent_memory_steps = save_recent_memory(step)
				break

		average_reward = calc_avgreward(total_reward)

		#monitor for stable performance. Stability is represented by more than 10 consecutive >180 rewards
		if consecutive_wins < 10:
			train_batch_size = dynamicbatchsize(total_reward, train_batch_size, len(memory), episode)
			trainwithmemory(train_batch_size, recent_memory, recent_memory_steps)
			if stable_flag and show_debug:
				print("Stability Lost at Episode {}".format(episode))
			stablity_episode = EPISODE
			stable_flag = 0
		elif not stable_flag: #if consecutive wins >= 10 and stable flag
			if show_debug:
				stablity_episode = episode
				print("Stability Achieved at Episode {}".format(episode))
			epsilon = FINAL_EPSILON
			stable_flag = 1
	    
		#print('episodes:', episode,'epsilon:',epsilon,
		#	'c_wins', consecutive_wins,
		#	'Reward:', total_reward,
		#	'avg_rwd', average_reward,
		#	'recent b:', step_list[0],
		#	'train b:', train_batch_size,
		#	'mem', len(memory),
		#	'stable_flag', stable_flag)

	    
	    # Test and view sample runs - can disable render to save time
		if (episode % TEST_FREQUENCY == 0 and episode != 0):
			total_reward = 0
			for i in range(TEST):
				state = env.reset()
				for j in range(STEP):
					#env.render()
					action = np.argmax(q_values.eval(feed_dict={
						state_in: [state]
					}))
					state, reward, done, _ = env.step(action)
					total_reward += reward
					if done:
						break
			ave_reward = total_reward / TEST
			print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
												'Average Reward:', ave_reward)

	    # Code below is used to prepare the scoreboard and measure performance
		if episode == 100:
			if ave_reward >= 190: 									
				run_scoreboard[x][0] = 1							
			else:
				failed_scores.append([x,ave_reward])
		if episode == 200 and ave_reward >= 190:	
			run_scoreboard[x][1] = 1								
	
	session.close()
	end_runtime = time.time()
	if stable_flag and consecutive_wins >= stability_requirement:		
		run_scoreboard[x][2] = 1										
	run_scoreboard[x][3] = stablity_episode								
	run_scoreboard[x][4] = round((end_runtime - start_runtime) / 60,2)	

env.close()
end_maintime = time.time()
total_runtime = round((end_maintime - start_maintime) / 60,2)

if show_debug:
	display_scoreboard(total_runtime)



