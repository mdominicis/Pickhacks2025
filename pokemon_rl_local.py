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
    def __init__(self, input_size, output_size, hidden_size=256):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.2)
        self.to(device)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_state = True
        else:
            single_state = False
            
        x = F.relu(self.fc1(x))
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        if x.size(0) > 1:
            x = self.bn3(x)
            
        x = self.fc4(x)
        
        return x.squeeze(0) if single_state else x

class PrioritizedReplayMemory:
    def __init__(self, capacity=10000, alpha=0.6, beta_start=0.4, beta_frames=10000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        
    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.pos] = (state, action, reward, next_state, done)
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.memory)]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones, indices, weights
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.memory)

TYPE_CHART = {
    "normal": {"rock": 0.5, "ghost": 0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 2, "bug": 2, "rock": 0.5, "dragon": 0.5, "steel": 2},
    "water": {"fire": 2, "water": 0.5, "grass": 0.5, "ground": 2, "rock": 2, "dragon": 0.5},
    "electric": {"water": 2, "electric": 0.5, "grass": 0.5, "ground": 0, "flying": 2, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2, "grass": 0.5, "poison": 0.5, "ground": 2, "flying": 0.5, "bug": 0.5, "rock": 2, "dragon": 0.5, "steel": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 0.5, "ground": 2, "flying": 2, "dragon": 2, "steel": 0.5},
    "fighting": {"normal": 2, "ice": 2, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2, "ghost": 0, "dark": 2, "steel": 2, "fairy": 0.5},
    "poison": {"grass": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0, "fairy": 2},
    "ground": {"fire": 2, "electric": 2, "grass": 0.5, "poison": 2, "flying": 0, "bug": 0.5, "rock": 2, "steel": 2},
    "flying": {"electric": 0.5, "grass": 2, "fighting": 2, "bug": 2, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5, "dark": 0, "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2, "ghost": 0.5, "dark": 2, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2, "ice": 2, "fighting": 0.5, "ground": 0.5, "flying": 2, "bug": 2, "steel": 0.5},
    "ghost": {"normal": 0, "psychic": 2, "ghost": 2, "dark": 0.5},
    "dragon": {"dragon": 2, "steel": 0.5, "fairy": 0},
    "dark": {"fighting": 0.5, "psychic": 2, "ghost": 2, "dark": 0.5, "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2, "rock": 2, "steel": 0.5, "fairy": 2},
    "fairy": {"fire": 0.5, "fighting": 2, "poison": 0.5, "dragon": 2, "dark": 2, "steel": 0.5}
}

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
        state_size=73, #was 128 when 128 it fucks up matrix multiplication 
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
        
        self.memory = PrioritizedReplayMemory()
        
        self.battles_played = 0
        self.target_update = target_update
        
        self.current_battle = None
        self.current_state = None
        self.last_action = None
        
        self.rewards = []
        self.wins = 0
        self.losses = 0
        
        self.last_active_hp_fraction = 1.0
        self.last_opponent_hp_fraction = 1.0
        self.last_opponent_status = None
        
        self.type_list = ["normal", "fire", "water", "electric", "grass", "ice", 
                          "fighting", "poison", "ground", "flying", "psychic", 
                          "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"]
        
        self.status_list = ["brn", "frz", "par", "psn", "slp", "tox"]
        self.weather_list = ["sun", "rain", "sand", "hail", "harsh_sunshine"]
        
    def calculate_type_effectiveness(self, move_type, defender_types):
        if not move_type or not defender_types:
            return 1.0
            
        effectiveness = 1.0
        #defender
        for def_type in defender_types:
            if def_type and move_type in TYPE_CHART and def_type in TYPE_CHART[move_type]:
                effectiveness *= TYPE_CHART[move_type][def_type] 
        
        return effectiveness
    
    def get_type_idx(self, type_name):
        try:
            return self.type_list.index(type_name)
        except (ValueError, TypeError):
            return -1
    
    def embed_battle(self, battle):
        features = []
        
        active_pokemon = battle.active_pokemon
        
        types_one_hot = [0] * 18
        if active_pokemon:
            for type_ in active_pokemon.types:
                if type_:
                    type_idx = self.get_type_idx(type_)
                    if type_idx >= 0:
                        types_one_hot[type_idx] = 1
        features.extend(types_one_hot)
        
        if active_pokemon:
            features.append(active_pokemon.current_hp / active_pokemon.max_hp)
        else:
            features.append(0)
        
        status_one_hot = [0] * 6
        if active_pokemon and active_pokemon.status:
            try:
                status_idx = self.status_list.index(active_pokemon.status)
                status_one_hot[status_idx] = 1
            except (ValueError, TypeError):
                pass
        features.extend(status_one_hot)
        
        opponent_active = battle.opponent_active_pokemon
        
        opp_types_one_hot = [0] * 18
        if opponent_active:
            for type_ in opponent_active.types:
                if type_:
                    type_idx = self.get_type_idx(type_)
                    if type_idx >= 0:
                        opp_types_one_hot[type_idx] = 1
        features.extend(opp_types_one_hot)
        
        if opponent_active and opponent_active.current_hp is not None:
            features.append(opponent_active.current_hp / opponent_active.max_hp)
        else:
            features.append(1.0)
        
        opp_status_one_hot = [0] * 6
        if opponent_active and opponent_active.status:
            try:
                status_idx = self.status_list.index(opponent_active.status)
                opp_status_one_hot[status_idx] = 1
            except (ValueError, TypeError):
                pass
        features.extend(opp_status_one_hot)
        
        features.append(len(battle.team) / 6)
        features.append(len(battle.opponent_team) / 6)
        
        for i in range(4):
            if i < len(battle.available_moves):
                move = battle.available_moves[i]
                
                power = min(move.base_power, 100) / 100
                
                effectiveness = 1.0
                if opponent_active and move.type:
                    effectiveness = self.calculate_type_effectiveness(move.type, opponent_active.types)
                effectiveness_norm = min(max((effectiveness - 0.25) / 3.75, 0), 1)
                
                accuracy = move.accuracy / 100 if move.accuracy else 1.0
                is_status = 1.0 if move.category == "status" else 0.0
                
                features.extend([power, effectiveness_norm, accuracy, is_status])
            else:
                features.extend([0, 0, 0, 0])
        
        weather_one_hot = [0] * 5
        if battle.weather:
            try:
                weather_idx = self.weather_list.index(battle.weather)
                weather_one_hot[weather_idx] = 1
            except (ValueError, TypeError):
                pass
        features.extend(weather_one_hot)
        
        return np.array(features, dtype=np.float32)
    
    def calc_reward(self, battle):
        reward = 0
        
        if battle.finished:
            if battle.won:
                reward += 10
                self.wins += 1
            else:
                reward -= 10
                self.losses += 1
            return reward
        
        current_active_hp_fraction = battle.active_pokemon.current_hp / battle.active_pokemon.max_hp if battle.active_pokemon else 0
        
        if hasattr(self, 'last_active_hp_fraction'):
            hp_change = current_active_hp_fraction - self.last_active_hp_fraction
            reward -= hp_change * 2
        self.last_active_hp_fraction = current_active_hp_fraction
        
        current_opponent_hp_fraction = battle.opponent_active_pokemon.current_hp / battle.opponent_active_pokemon.max_hp if battle.opponent_active_pokemon and battle.opponent_active_pokemon.current_hp is not None else 1.0
        
        if hasattr(self, 'last_opponent_hp_fraction'):
            opp_hp_change = current_opponent_hp_fraction - self.last_opponent_hp_fraction
            reward -= opp_hp_change * 2
        self.last_opponent_hp_fraction = current_opponent_hp_fraction
        
        if battle.opponent_active_pokemon and battle.opponent_active_pokemon.fainted:
            reward += 2
        if battle.active_pokemon and battle.active_pokemon.fainted:
            reward -= 2
        
        status_value = {
            'brn': 0.5, 'frz': 1.0, 'par': 0.5, 
            'psn': 0.3, 'slp': 0.8, 'tox': 0.7
        }
        if battle.opponent_active_pokemon and battle.opponent_active_pokemon.status:
            if not hasattr(self, 'last_opponent_status') or self.last_opponent_status != battle.opponent_active_pokemon.status:
                reward += status_value.get(battle.opponent_active_pokemon.status, 0.3)
        self.last_opponent_status = battle.opponent_active_pokemon.status if battle.opponent_active_pokemon else None
        
        return reward
    
    def _should_switch(self, battle):
        if not battle.available_switches:
            return False
        
        active_pokemon = battle.active_pokemon
        opponent_active = battle.opponent_active_pokemon
        
        if active_pokemon and opponent_active:
            worst_matchup = 1.0
            for opponent_type in opponent_active.types:
                if not opponent_type:
                    continue
                for my_type in active_pokemon.types:
                    if not my_type:
                        continue
                    effectiveness = self._get_defensive_effectiveness(my_type, opponent_type)
                    worst_matchup = min(worst_matchup, effectiveness)
            
            if worst_matchup <= 0.25:
                return random.random() < 0.8
            elif worst_matchup <= 0.5:
                return random.random() < 0.5
        
        if active_pokemon and active_pokemon.current_hp / active_pokemon.max_hp < 0.25:
            return random.random() < 0.7
        
        if active_pokemon and active_pokemon.status:
            return random.random() < 0.4
        
        return False

    def _choose_switch(self, battle):
        switches = battle.available_switches
        
        switch_scores = []
        opponent_active = battle.opponent_active_pokemon
        
        for pokemon in switches:
            score = 0
            
            score += pokemon.current_hp / pokemon.max_hp
            
            if opponent_active:
                for my_type in pokemon.types:
                    if not my_type:
                        continue
                    for opponent_type in opponent_active.types:
                        if not opponent_type:
                            continue
                        effectiveness = self._get_offensive_effectiveness(my_type, opponent_type)
                        score += (effectiveness - 1) * 0.5
            
            if pokemon.status:
                status_penalty = {
                    'brn': 0.3, 'frz': 0.8, 'par': 0.4, 
                    'psn': 0.2, 'slp': 0.6, 'tox': 0.5
                }
                score -= status_penalty.get(pokemon.status, 0.3)
            
            switch_scores.append((pokemon, score))
        
        best_switch = max(switch_scores, key=lambda x: x[1])[0]
        return self.create_order(best_switch)

    def _get_offensive_effectiveness(self, attacking_type, defending_type):
        if attacking_type in TYPE_CHART and defending_type in TYPE_CHART[attacking_type]:
            return TYPE_CHART[attacking_type][defending_type]
        return 1.0

    def _get_defensive_effectiveness(self, defending_type, attacking_type):
        if attacking_type in TYPE_CHART and defending_type in TYPE_CHART[attacking_type]:
            return TYPE_CHART[attacking_type][defending_type]
        return 1.0
    
    def choose_move(self, battle):

        if battle.finished:
          return self.choose_random_move(battle)
        if self._should_switch(battle):
            return self._choose_switch(battle)
            
        self.current_battle = battle
        state = self.embed_battle(battle)
        self.current_state = state
        available_moves = battle.available_moves

        if not available_moves:
            return self.choose_random_move(battle)

        if self.training and random.random() < self.epsilon:
            move_scores = []
            for move in available_moves:
                base_score = min(move.base_power, 100)
                
                if battle.opponent_active_pokemon and move.type:
                    effectiveness = self.calculate_type_effectiveness(move.type, battle.opponent_active_pokemon.types)
                    base_score *= effectiveness
                
                if move.category == "status" and random.random() < 0.3:
                    base_score += 40
                    
                base_score = max(base_score, 0.1)
                move_scores.append(base_score)
                
            total_score = sum(move_scores) or 1
            probabilities = [score/total_score for score in move_scores]

            sum_prob = sum(probabilities)
            if sum_prob > 0:
                probabilities = [p/sum_prob for p in probabilities]
            else:
                # If all probabilities are zero (shouldn't)
                probabilities = [1.0/len(probabilities) for _ in probabilities]

            move_idx = np.random.choice(len(available_moves), p=probabilities)
            move = available_moves[move_idx]
            action_idx = move_idx
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(np.array([state], dtype=np.float32), dtype=torch.float).to(device)
                q_values = self.policy_net(state_tensor)

                mask = torch.ones(self.action_size, device=device) * -1e9
                for i, move in enumerate(available_moves):
                    mask[i] = 0
                q_values = q_values + mask

                action_idx = q_values.max(1)[1].item()
                if action_idx < len(available_moves):
                    move = available_moves[action_idx]
                else:
                    move_scores = [(m, min(m.base_power, 100) * 
                                  self.calculate_type_effectiveness(m.type, battle.opponent_active_pokemon.types) 
                                  if battle.opponent_active_pokemon else min(m.base_power, 100)) 
                                  for m in available_moves]
                    move = max(move_scores, key=lambda x: x[1])[0]
                    action_idx = available_moves.index(move)

        self.last_action = action_idx

        if self.training and self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return self.create_order(move)
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        
        state_batch = torch.tensor(np.array(states, dtype=np.float32), dtype=torch.float).to(device)
        action_batch = torch.tensor(np.array(actions, dtype=np.int64), dtype=torch.long).unsqueeze(1).to(device)
        reward_batch = torch.tensor(np.array(rewards, dtype=np.float32), dtype=torch.float).to(device)
        next_state_batch = torch.tensor(np.array(next_states, dtype=np.float32), dtype=torch.float).to(device)
        done_batch = torch.tensor(np.array(dones, dtype=np.bool_), dtype=torch.bool).to(device)
        weights_batch = torch.tensor(np.array(weights, dtype=np.float32), dtype=torch.float).to(device)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
        
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        expected_q_values = reward_batch + self.gamma * next_q_values * (~done_batch)
        
        loss = F.smooth_l1_loss(q_values, expected_q_values, reduction='none')
        loss = (loss * weights_batch).mean()
        
        with torch.no_grad():
            td_errors = torch.abs(expected_q_values - q_values).cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-5)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
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
        self.last_active_hp_fraction = 1.0
        self.last_opponent_hp_fraction = 1.0
        self.last_opponent_status = None
    
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