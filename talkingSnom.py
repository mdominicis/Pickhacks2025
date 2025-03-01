import asyncio
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import RandomPlayer

async def main():
    player = RandomPlayer(
        account_configuration=AccountConfiguration("talking snom", "pickhacks2025"),
        )

    await player.send_challenges("Elytrix", n_challenges=1)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())