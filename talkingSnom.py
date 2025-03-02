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
    async def trash_talk(self,battle):
        last_turn = battle.observations[battle._turn-1]
        end_prompt = ". Do this in one paragraph and only use dialogue, do not describe your actions. do not speak out of character."
        if battle._turn == 1:
            #FIRST TURN MESSAGE
            message = chat.send_message("Pretend you are a Pokemon trainer. Challenge me to a pokemon battle. Your first pokemon is "+battle.active_pokemon._name+end_prompt)
        elif battle.opponent_active_pokemon.is_terastallized != last_turn.opponent_active_pokemon.is_terastallized:
            #HUMAN TERA
            message = chat.send_message("Uh oh, your opponent just terastallized! This changes their pokemon's type. say your reaction"+end_prompt)
        else:
            return 0
        await self.ps_client.send_message(message.text, battle.battle_tag)


async def main():
    #Uses my testing account
    player = MaxDamagePlayer(
        account_configuration=AccountConfiguration("talking snom", "pickhacks2025"),
        )

    #Challenges me
    await player.send_challenges("Elytrix", n_challenges=1)



if __name__ == "__main__":
    #Sets api key and creates a "chat" which trash_talk will use
    client = genai.Client(api_key="AIzaSyDY50hm-IIwxEgH9hvNEk9ews7Het3v1Qw")
    chat = client.chats.create(model="gemini-2.0-flash")
    
    asyncio.get_event_loop().run_until_complete(main())