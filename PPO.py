import torch
import Actor_Critic
import EnvironmentManager
import Drive
import Bar_graph
import GAE
import Agent
import Memory

# Hyper parameters:
lr = 0.00001
steps = 2500
gamma = 0.99
tau = 0.95
mini_batch = 128
Epochs = 15
epsilon = 0.4

path = 'G:/My Drive/models/'
model_name = 'PPO_Packman.mdl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

drive_ = Drive.model_drive(path, model_name)
model_exists = drive_.does_exist()

player_model = Actor_Critic.Actor_Critic(device, 3).to(device)
ghost_model = Actor_Critic.Actor_Critic(device, 3).to(device)
if model_exists:
    checkpoint = drive_.download()
    Main_episode = checkpoint['Main_episode']
    High_score = checkpoint['High_score']
    player_model.load_state_dict(checkpoint['player_state_dict'])
    ghost_model.load_state_dict(checkpoint['ghost_state_dict'])
else:
    checkpoint = None
    High_score = 0
    Main_episode = 1

gae = GAE.GAE(gamma, tau, device)
Player_agent = Agent.Agent_player(lr, player_model, epsilon, gae, checkpoint)
Ghost_agent = Agent.Agent_ghost(lr, ghost_model, epsilon, gae, checkpoint)
env = EnvironmentManager.EnvManager(device)

player_memory = Memory.memory(device)
ghost_memory = Memory.memory(device)

Bar_log = Bar_graph.bar_update()

Game = 1
Update = 100
points_all = 0
loss = 0

while True:

    points = 0
    new_points = 0

    for timestep in range(steps):

        player_state, player_direction, ghost_direction, hunter_state = env.get_state()
        player_value, player_dist = Player_agent.get_action(player_state, player_direction, hunter_state)
        player_action = player_dist.sample()
        player_reward, new_points, end_state = env.take_action_player(player_action.item())
        player_log_prob = player_dist.log_prob(player_action)

        if end_state:
            env.reset()
            temp = torch.zeros((1, 1), dtype=torch.float64).to(device)
            player_memory.add_to_memory(player_state, player_direction, hunter_state, player_action, temp, player_reward, 0, player_log_prob)
            ghost_memory.rewards[-1] = 100
            ghost_memory.masks[-1] = 0
            break

        player_memory.add_to_memory(player_state, player_direction, hunter_state, player_action, player_value, player_reward, 1, player_log_prob)

        ghost_state, player_direction, ghost_direction, hunter_state = env.get_state()
        ghost_value, ghost_dist = Ghost_agent.get_action(ghost_state, ghost_direction, hunter_state)
        ghost_action = ghost_dist.sample()
        ghost_reward, end_state = env.take_action_ghost(ghost_action.item())
        ghost_log_prob = ghost_dist.log_prob(ghost_action)

        if end_state:
            env.reset()
            ghost_memory.add_to_memory(ghost_state, ghost_direction, hunter_state, ghost_action, ghost_value, ghost_reward, 0, ghost_log_prob)
            player_memory.rewards[-1] = -100
            player_memory.masks[-1] = 0
            break

        ghost_memory.add_to_memory(ghost_state, ghost_direction, hunter_state, ghost_action, ghost_value, ghost_reward, 1, ghost_log_prob)

    env.reset()

    if High_score < new_points:
        High_score = new_points

    player_returns = Player_agent.get_returns(player_memory.rewards, player_memory.values, player_memory.masks)
    ghost_returns = Ghost_agent.get_returns(ghost_memory.rewards, ghost_memory.values, ghost_memory.masks)

    states, directions, hunter_states, actions, log_probs, returns, advantages = player_memory.make_data(player_returns) 
    loss += Player_agent.train(Epochs, mini_batch, states, directions, hunter_states, actions, log_probs, returns, advantages)

    states, directions, hunter_states, actions, log_probs, returns, advantages = ghost_memory.make_data(ghost_returns)
    Ghost_agent.train(Epochs, mini_batch, states, directions, hunter_states, actions, log_probs, returns, advantages)

    points_all += new_points
    Bar_log.print_info(Game, loss / Game / Epochs, points_all / Game, High_score, Update, Main_episode)

    Game += 1

    player_memory.reset()
    ghost_memory.reset()

    torch.cuda.empty_cache()

    if Game - 1 == Update:

        drive_.upload(Player_agent.model, Ghost_agent.model, Player_agent.optimizer, Ghost_agent.optimizer, Main_episode, High_score)

        Bar_log.new_bar()

        Main_episode += 1
        points_all = 0
        loss = 0
        Game = 1


