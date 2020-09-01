import neat
import pickle
from traingame.player import NeatSpeedAI
from traingame.game import Environment, Engine


def eval_genomes(genomes, config):
    ai_players = []
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        ai_players.append(NeatSpeedAI(net))

    track = Environment("assen")
    max_score = track.distance_matrix.max()
    game_engine = Engine(
        headless=False,
        environment=track,
        players=ai_players,
        tick_limit=max_score * 2
    )
    game_engine.play()
    scores = game_engine.get_scores()
    for idx, (genome_id, genome) in enumerate(genomes):
        try:
            life_span = game_engine.players[idx].ending_tick - game_engine.players[idx].starting_tick
        except AttributeError:
            life_span = game_engine.tick - game_engine.players[idx].starting_tick
        genome.fitness = (max_score - scores[idx]) - (life_span - (max_score / 2.) / 1.)


if __name__ == "__main__":
    # load configuration
    configuration = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "example-config.cfg"
    )
    p = neat.Population(configuration)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    best_model = p.run(eval_genomes)

    with open("best_model.pickle", "wb") as f:
        pickle.dump(best_model, f)
