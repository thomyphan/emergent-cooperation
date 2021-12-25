import mate.environments.matrix_games as matrix_games
import mate.environments.coin_game as coin_game
import mate.environments.harvest as harvest

def make(params):
    domain_name = params["domain_name"]
    if domain_name.startswith("Matrix-"):
        params["R_max"] = 3
        return matrix_games.make(params)
    if domain_name.startswith("CoinGame-"):
        params["R_max"] = 2
        return coin_game.make(params)
    if domain_name.startswith("Harvest-"):
        params["R_max"] = 0.25
        return harvest.make(params)
    raise ValueError("Unknown domain '{}'".format(domain_name))