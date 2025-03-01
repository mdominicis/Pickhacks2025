import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
import random
import asyncio
import os
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env import AccountConfiguration, ServerConfiguration


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")

#seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

os.environ["POKE_ENV_SERVER_URL"] = "http://localhost:8000"
os.environ["POKE_ENV_SERVER_HOST"] = "localhost"
os.environ["POKE_ENV_SERVER_PORT"] = "8000"

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.to(device)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.memory)

class PokemonRLPlayer(Player):
    def __init__(
        self,
        battle_format="gen9randombattle",
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        target_update=10,
        state_size=44,
        training=True
    ):
        super().__init__(
            battle_format=battle_format
        )
        
        self.training = training
        self.state_size = state_size
        self.action_size = 4
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.policy_net = DQNNetwork(state_size, self.action_size)
        self.target_net = DQNNetwork(state_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        self.memory = ReplayMemory()
        
        self.battles_played = 0
        self.target_update = target_update
        
        self.current_battle = None
        self.current_state = None
        self.last_action = None
        
        self.rewards = []
        self.wins = 0
        self.losses = 0
        
    def embed_battle(self, battle):
        active_pokemon = battle.active_pokemon
        
        features = []
        
        types_one_hot = [0] * 18
        if active_pokemon:
            for type_ in active_pokemon.types:
                if type_:
                    type_idx = hash(type_) % 18
                    types_one_hot[type_idx] = 1
        features.extend(types_one_hot)
        
        if active_pokemon:
            features.append(active_pokemon.current_hp / active_pokemon.max_hp)
        else:
            features.append(0)
        
        opponent_active = battle.opponent_active_pokemon
        
        opp_types_one_hot = [0] * 18
        if opponent_active:
            for type_ in opponent_active.types:
                if type_:
                    type_idx = hash(type_) % 18
                    opp_types_one_hot[type_idx] = 1
        features.extend(opp_types_one_hot)
        
        if opponent_active and opponent_active.current_hp is not None:
            features.append(opponent_active.current_hp / opponent_active.max_hp)
        else:
            features.append(1.0)
            
        features.append(len(battle.team) / 6)
        features.append(len(battle.opponent_team) / 6)
        
        for i in range(4):
            if i < len(battle.available_moves):
                move = battle.available_moves[i]
                features.append(min(move.base_power, 100) / 100)
            else:
                features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def calc_reward(self, battle):
        reward = 0
        
        if battle.finished:
            if battle.won:
                reward = 10
                self.wins += 1
            else:
                reward = -10
                self.losses += 1
        else:
            fainted_mon_reward = -0.5 if battle.active_pokemon.fainted else 0
            
            if battle.opponent_active_pokemon.fainted:
                reward += 1
            
            reward += fainted_mon_reward
        
        return reward
    
    def choose_move(self, battle): 
        self.current_battle = battle

        state = self.embed_battle(battle)

        self.current_state = state

        available_moves = battle.available_moves

        if not available_moves:
            return self.choose_random_move(battle)

        if self.training and random.random() < self.epsilon:
            move = max(available_moves, key=lambda move: move.base_power) #change to be better lel
            action_idx = available_moves.index(move)
        else:
            with torch.no_grad():
                state_np = np.array([state], dtype=np.float32)
                state_tensor = torch.tensor(state_np, dtype=torch.float).to(device)
                q_values = self.policy_net(state_tensor)

                mask = torch.ones(self.action_size, device=device) * -1e9
                for i, move in enumerate(available_moves):
                    move_idx = i
                    mask[move_idx] = 0

                q_values = q_values + mask

                action_idx = q_values.max(1)[1].item()
                if action_idx < len(available_moves):
                    move = available_moves[action_idx]
                else:
                    move = max(available_moves, key=lambda move: move.base_power)
                    action_idx = available_moves.index(move)

        self.last_action = action_idx

        if self.training and self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return self.create_order(move)
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        state_batch = torch.tensor(np.array(states, dtype=np.float32), dtype=torch.float).to(device)
        action_batch = torch.tensor(np.array(actions, dtype=np.int64), dtype=torch.long).unsqueeze(1).to(device)
        reward_batch = torch.tensor(np.array(rewards, dtype=np.float32), dtype=torch.float).to(device)
        next_state_batch = torch.tensor(np.array(next_states, dtype=np.float32), dtype=torch.float).to(device)
        done_batch = torch.tensor(np.array(dones, dtype=np.bool_), dtype=torch.bool).to(device)
        
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_q_values = torch.zeros(self.batch_size, dtype=torch.float, device=device)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        
        expected_q_values = reward_batch + (self.gamma * next_q_values * (~done_batch))
        
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    async def battle_end_callback(self, battle):
        reward = self.calc_reward(battle)
        self.rewards.append(reward)
        
        if self.current_state is not None and self.last_action is not None:
            done = True
            next_state = self.current_state
            self.memory.push(self.current_state, self.last_action, reward, next_state, done)
        
        if self.training:
            self.learn()
            
        self.battles_played += 1
        
        if self.battles_played % self.target_update == 0:
            self.update_target_network()
            
        if self.battles_played % 10 == 0:
            win_rate = self.wins / self.battles_played * 100
            print(f"Battles: {self.battles_played}, Win rate: {win_rate:.2f}%, Epsilon: {self.epsilon:.2f}")
        
        self.current_state = None
        self.last_action = None
    
    async def battle_callback(self, battle):
        if self.current_state is not None and self.last_action is not None:
            reward = self.calc_reward(battle)
            
            next_state = self.embed_battle(battle)
            
            done = False
            self.memory.push(self.current_state, self.last_action, reward, next_state, done)
            
            if self.training:
                self.learn()

def save_model(model, path="pokemon_rl_model.pt"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="pokemon_rl_model.pt"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded from {path}")
        return True
    return False

async def train_model(num_battles=100):
    rl_player = PokemonRLPlayer(
        battle_format="gen9randombattle", 
        training=True
    )
    
    model_exists = load_model(rl_player.policy_net)
    if model_exists:
        print("Continuing training with existing model")
    else:
        print("Starting fresh training")
        
    max_damage_opponent = MaxDamagePlayer(
        battle_format="gen9randombattle"
    )
    
    print(f"Connecting to local Pokémon Showdown server as configured in environment variables")
    
    print(f"Starting training for {num_battles} battles against MaxDamagePlayer...")
    
    for i in range(0, num_battles, 10):
        await max_damage_opponent.battle_against(rl_player, n_battles=10)
        
        if rl_player.battles_played > 0:
            win_rate = rl_player.wins / rl_player.battles_played * 100
            print(f"Completed {rl_player.battles_played} battles, Win rate: {win_rate:.2f}%, Epsilon: {rl_player.epsilon:.2f}")
        else:
            print("No battles completed yet")
    
    save_model(rl_player.policy_net)
    
    return rl_player

async def evaluate_model(num_battles=50):
    rl_player = PokemonRLPlayer(
        battle_format="gen9randombattle", 
        training=False, 
        epsilon_start=0.05
    )
    
    load_model(rl_player.policy_net)
    
    rl_player.wins = 0
    rl_player.losses = 0
    rl_player.battles_played = 0
    
    random_opponent = RandomPlayer(
        battle_format="gen9randombattle"
    )
    
    max_damage_opponent = MaxDamagePlayer(
        battle_format="gen9randombattle"
    )
    
    print("Evaluating against Random Player...")
    await random_opponent.battle_against(rl_player, n_battles=num_battles//2)
    random_win_rate = rl_player.wins / (num_battles//2) * 100
    print(f"Win rate against Random Player: {random_win_rate:.2f}%")
    
    rl_player.wins = 0
    rl_player.losses = 0
    rl_player.battles_played = 0
    
    print("Evaluating against Max Damage Player...")
    await max_damage_opponent.battle_against(rl_player, n_battles=num_battles//2)
    max_damage_win_rate = rl_player.wins / (num_battles//2) * 100
    print(f"Win rate against Max Damage Player: {max_damage_win_rate:.2f}%")


async def main():
    print("Using local Pokémon Showdown server via environment variables")
    print("Make sure your server is running with: node pokemon-showdown start --no-security")
    
    rl_player = await train_model(num_battles=100)

    await evaluate_model(num_battles=50)

if __name__ == "__main__":
    asyncio.run(main())