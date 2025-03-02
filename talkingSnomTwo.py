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
        #observation data from last turn
        last_turn = battle.observations[battle._turn-1]
        #used for ai response sharpening
        end_prompt = ". Your current pokemon is "+battle.active_pokemon.species+". Do this in one paragraph and only use dialogue, do not describe your actions. do not speak out of character. do not say your pokemon's attacks."
        if battle._turn!=1:
            #init oppFnt
            oppFnt = False
            #stores if this is the same pokemon as last turn (opponent)
            if (last_turn.opponent_active_pokemon.species == battle.opponent_active_pokemon.species):
                oppSwitch = False
            else:
                oppSwitch = True
            #stores if this is the same pokemon as last turn (ally)
            if (last_turn.active_pokemon.species == battle.active_pokemon.species):
                allySwitch = False
            else:
                allySwitch = True
            #stores if the opponent's pokemon fainted
            for key in battle.opponent_team:
                if battle.opponent_team[key].species == last_turn.opponent_active_pokemon.species and str(battle.opponent_team[key].status) == "FNT (status) object":
                    oppFnt = True


        #conditional for choosing most important event
        if battle._turn == 1 and str(battle.active_pokemon.status)!="FNT (status) object":
            #FIRST TURN MESSAGE
            message = chat.send_message("Pretend you are a Pokemon trainer. Challenge me to a pokemon battle. Your first pokemon is "+battle.active_pokemon._name+end_prompt)
        elif oppFnt == True:
            #OPPONENT FAINT
             message = chat.send_message("Yes! Your opponent's pokemon, "+last_turn.opponent_active_pokemon.species+" just fainted! But the battle isn't done just yet!"+end_prompt)
        elif str(battle.active_pokemon.status) == "FNT (status) object":
            #ALLY FAINT
             message = chat.send_message("Oh no, "+battle.active_pokemon.species+" just fainted! But the battle isn't done just yet! Do not say the name of your next pokemon."+end_prompt)
        elif battle.opponent_active_pokemon.is_terastallized != last_turn.opponent_active_pokemon.is_terastallized:
            #OPPONENT TERA
            message = chat.send_message("Uh oh, your opponent just terastallized! This changes their pokemon's type. say your reaction"+end_prompt)
        elif battle.active_pokemon.is_terastallized != last_turn.active_pokemon.is_terastallized and oppSwitch == False:
            #ALLY TERA
            message = chat.send_message("You just terastallized your pokemon, "+battle.active_pokemon+"! This changes your pokemon's type. say your reaction"+end_prompt)
        elif oppSwitch == True:
            #OPPONENT SWITCH
            message = chat.send_message("your opponent sent out their pokemon "+battle.opponent_active_pokemon.species+"! describe your reaction"+end_prompt)
        elif allySwitch == True:
            #ALLY SWITCH
            message = chat.send_message("you sent out your pokemon "+battle.active_pokemon.species+"! describe your reaction"+end_prompt)
        elif battle.opponent_active_pokemon.status != last_turn.opponent_active_pokemon.status and oppSwitch == False:
            #OPPONENT STATUS
            match str(battle.opponent_active_pokemon.status):
                case "BRN (status) object":
                    status = "burn"
                case "FRZ (status) object":
                    status = "frozen"
                case "PAR (status) object":
                    status = "paralysis"
                case "SLP (status) object":
                    status = "sleep"
                case "TOX (status) object":
                    status = "badly poisoned"
                case _:
                    return 0
            message = chat.send_message("The opponent's pokemon "+battle.opponent_active_pokemon.species+" just got afflicted with the "+status+" status effect! describe your reaction"+end_prompt)

        elif battle.active_pokemon.status != last_turn.active_pokemon.status and allySwitch == False:
            #ALLY STATUS
            match str(battle.active_pokemon.status):
                case "BRN (status) object":
                    status = "burn"
                case "FRZ (status) object":
                    status = "frozen"
                case "PAR (status) object":
                    status = "paralysis"
                case "SLP (status) object":
                    status = "sleep"
                case "TOX (status) object":
                    status = "badly poisoned"
                case _:
                    return 0
            message = chat.send_message("Oh no! your pokemon "+battle.active_pokemon.species+" just got afflicted with the "+status+" status effect! describe your reaction"+end_prompt)
        else:
            return 0
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
