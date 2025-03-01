import random
import os
import asyncio
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env import AccountConfiguration, ServerConfiguration

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

class RuleBasedPokemonAI(Player):
    type_chart = {
        'normal': {'weak': ['fighting'], 'strong': [], 'immune': ['ghost']},
        'fire': {'weak': ['water', 'ground', 'rock'], 'strong': ['grass', 'ice', 'bug', 'steel'], 'immune': []},
        'water': {'weak': ['grass', 'electric'], 'strong': ['fire', 'ground', 'rock'], 'immune': []},
        'grass': {'weak': ['fire', 'ice', 'poison', 'flying', 'bug'], 'strong': ['water', 'ground', 'rock'], 'immune': []},
        'electric': {'weak': ['ground'], 'strong': ['water', 'flying'], 'immune': []},
        'ice': {'weak': ['fire', 'fighting', 'rock', 'steel'], 'strong': ['grass', 'ground', 'flying', 'dragon'], 'immune': []},
        'fighting': {'weak': ['flying', 'psychic', 'fairy'], 'strong': ['normal', 'rock', 'steel', 'ice', 'dark'], 'immune': []},
        'poison': {'weak': ['ground', 'psychic'], 'strong': ['grass', 'fairy'], 'immune': []},
        'ground': {'weak': ['water', 'grass', 'ice'], 'strong': ['fire', 'electric', 'poison', 'rock', 'steel'], 'immune': ['electric']},
        'flying': {'weak': ['electric', 'ice', 'rock'], 'strong': ['grass', 'fighting', 'bug'], 'immune': ['ground']},
        'psychic': {'weak': ['bug', 'dark', 'ghost'], 'strong': ['fighting', 'poison'], 'immune': []},
        'bug': {'weak': ['fire', 'flying', 'rock'], 'strong': ['grass', 'psychic', 'dark'], 'immune': []},
        'rock': {'weak': ['water', 'grass', 'fighting', 'ground', 'steel'], 'strong': ['fire', 'ice', 'flying', 'bug'], 'immune': []},
        'ghost': {'weak': ['ghost', 'dark'], 'strong': ['psychic', 'ghost'], 'immune': ['normal', 'fighting']},
        'dragon': {'weak': ['ice', 'dragon', 'fairy'], 'strong': ['dragon'], 'immune': []},
        'dark': {'weak': ['fighting', 'bug', 'fairy'], 'strong': ['psychic', 'ghost'], 'immune': []},
        'steel': {'weak': ['fire', 'fighting', 'ground'], 'strong': ['ice', 'rock', 'fairy'], 'immune': ['poison']},
        'fairy': {'weak': ['poison', 'steel'], 'strong': ['fighting', 'dragon', 'dark'], 'immune': ['dragon']}
    }

    def calculate_damage_multiplier(self, attack_type, defense_types):
        multiplier = 1.0
        for defense_type in defense_types:
            if defense_type in self.type_chart[attack_type]['strong']:
                multiplier *= 2.0
            elif defense_type in self.type_chart[attack_type]['weak']:
                multiplier *= 0.5
            elif defense_type in self.type_chart[attack_type]['immune']:
                multiplier *= 0.0
        return multiplier

    # Choose Highest Damage move based on type
    def choose_attack_move(self, battle):
        best_move = None
        best_score = -1

        for move in battle.available_moves:
            move_type = move.type.name.lower()
            opponent_types = [t.name.lower() for t in battle.opponent_active_pokemon.types]
            effectiveness = self.calculate_damage_multiplier(move_type, opponent_types)
            score = move.base_power * effectiveness

            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move:
            return self.create_order(best_move)
        
        # If no attacking moves, switch to another Pokémon
        if battle.available_switches:
            return self.create_order(random.choice(battle.available_switches))
        
        # Default
        return self.choose_random_move(battle)

    # Switch if low    
    def switch_pokemon(self,battle):
        if battle.available_switches:
            opponent_types = [t.name.lower() for t in battle.opponent_active_pokemon.types]
            best_switch = None
            best_score = -1

            for switch in battle.available_switches:
                switch_types = [t.name.lower() for t in switch.types]
                score = sum(self.calculate_damage_multiplier(opp_type, switch_types) for opp_type in opponent_types)
                
                if score == 0:  # Immune
                    return self.create_order(switch)
                
                if score < best_score:
                    best_score = score
                    best_switch = switch
            
            if best_switch:
                return self.create_order(best_switch)
        
        return self.choose_random_move(battle)

    def choose_move(self,battle):
        current_pokemon_types = [t.name.lower() for t in battle.active_pokemon.types]
        opponent_types = [t.name.lower() for t in battle.opponent_active_pokemon.types]
        is_weak = False

        for type in current_pokemon_types:
            if any(weak_type in self.type_chart[type]['weak'] for weak_type in opponent_types):
                is_weak = True
                break

        if is_weak:
            return self.switch_pokemon(battle)
        if battle.available_moves:
            return self.choose_attack_move(battle)
        
        # Fallback to switching if no moves are available
        return self.switch_pokemon(battle)


# Server
server_config = ServerConfiguration("localhost", 8000)
player1_config = AccountConfiguration("rule_based_ai", "password")
player2_config = AccountConfiguration("max_damage_ai", "password")

# Initialize the players
rule_based_ai = RuleBasedPokemonAI(player1_config)
max_damage_ai = MaxDamagePlayer(player2_config)

# Create a battle and have both AIs play against each other
async def run_battle():
    battle = await rule_based_ai.battle_against(max_damage_ai, n_battles=10)


if __name__ == "__main__":
    asyncio.run(run_battle())