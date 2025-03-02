import asyncio
from google import genai
import poke_env
import random
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import Player
from typing import Awaitable

class MaxDamagePlayer(Player):

    #IGNORE THIS    
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

    #Function changed to include trash_talk
    async def _handle_battle_request(
        self,
        battle,
        from_teampreview_request: bool = False,
        maybe_default_order: bool = False,
    ):
        #IMPORTANT: COMMAND BELOW IS ADDED TO FUNCTION
        await self.trash_talk(battle)
        if maybe_default_order and random.random() < self.DEFAULT_CHOICE_CHANCE:
            message = self.choose_default_move().message
        elif battle.teampreview:
            if not from_teampreview_request:
                return
            message = self.teampreview(battle)
        else:
            choice = self.choose_move(battle)
            if isinstance(choice, Awaitable):
                choice = await choice
            message = choice.message

        if not battle._wait:
            await self.ps_client.send_message(message, battle.battle_tag)

    #Basic trash talk
    async def trash_talk(self, battle):
        last_turn = battle.observations[battle._turn - 1]
        end_prompt = ". Do this in one paragraph and only use dialogue, do not describe your actions. Do not speak out of character."

        current_active_pokemon = battle.active_pokemon._name
        current_opponent_pokemon = battle.opponent_active_pokemon._name

        message = None

        # First turn message
        if battle._turn == 1:
            message = chat.send_message(
                f"Pretend you are a Pokémon trainer. Challenge me to a Pokémon battle. Your first Pokémon is {current_active_pokemon}{end_prompt}"
            )

        # Opponent Terastallized
        elif battle.opponent_active_pokemon.is_terastallized != last_turn.opponent_active_pokemon.is_terastallized:
            message = chat.send_message(
                f"Uh oh, your opponent just Terastallized! This changes their Pokémon's type. Say your reaction{end_prompt}"
            )

        # Check if player's Pokémon changed
        elif self.last_active_pokemon != current_active_pokemon:
            message = chat.send_message(
                f"You switched to {current_active_pokemon}. Say something confident about your new Pokémon!{end_prompt}"
            )

        # Check if opponent's Pokémon changed
        elif self.last_opponent_pokemon != current_opponent_pokemon:
            message = chat.send_message(
                f"Your opponent switched to {current_opponent_pokemon}. Say something snarky about it!{end_prompt}"
            )

        # Update last seen Pokémon
        self.last_active_pokemon = current_active_pokemon
        self.last_opponent_pokemon = current_opponent_pokemon

        # Send message if there is one
        if message:
            await self.ps_client.send_message(message.text, battle.battle_tag)


async def main():
    #Uses my testing account
    player = MaxDamagePlayer(
        account_configuration=AccountConfiguration("AIvysaur_xx xx_X_", "pickhacks2025"),
        )

    #Challenges me
    await player.send_challenges("mdominicis", n_challenges=1)



if __name__ == "__main__":
    #Sets api key and creates a "chat" which trash_talk will use
    client = genai.Client(api_key="")
    chat = client.chats.create(model="gemini-2.0-flash")
    
    asyncio.get_event_loop().run_until_complete(main())