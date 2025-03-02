import asyncio
from google import genai
import poke_env
import random
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import Player
from typing import Awaitable

class MaxDamagePlayer(Player):    
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

    async def _handle_battle_request(
        self,
        battle,
        from_teampreview_request: bool = False,
        maybe_default_order: bool = False,
    ):
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

    async def trash_talk(self,battle):
        print("fish")
        message = chat.send_message("I have 2 dogs in my house.")
        await self.ps_client.send_message(message.text, battle.battle_tag)


async def main():
    player = MaxDamagePlayer(
        account_configuration=AccountConfiguration("talking snom", "pickhacks2025"),
        )

    await player.send_challenges("Elytrix", n_challenges=1)



if __name__ == "__main__":
    client = genai.Client(api_key="AIzaSyALnRlIglNFd5kEhJadfP_rD4Qmmx45qmo")
    chat = client.chats.create(model="gemini-2.0-flash")
    asyncio.get_event_loop().run_until_complete(main())