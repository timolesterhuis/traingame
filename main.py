from traingame.game import Engine, Environment
from traingame.player import HumanPlayer


def main():
    track = Environment('assen')

    game_engine = Engine(
        environment=track,
        players=[HumanPlayer()]
    )

    game_engine.play()


if __name__ == "__main__":
    main()
