#!/usr/bin/env python3
from misio.optilio.pacman import StdIOPacmanRunner
from misio.pacman.pacman import LocalPacmanGameRunner
from misio.util import generate_deterministic_seeds
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from pacquiao_agent import PacQuiaoAgent
import numpy as np
import random
import tqdm

def parse_args():
    parser = ArgumentParser(description="Pacman runner program.", formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-n", "--num-games", type=int, default=500,
                        metavar="NUM_GAMES",
                        help="the number of GAMES to play")
    parser.add_argument("-l", "--layout", metavar="LAYOUT_FILE", default="pacman_layouts/mediumClassic.lay",
                        help="the LAYOUT_FILE from which to load the map layout (see pacman_layouts directory)"
                        )
    parser.add_argument("-sg", "--smart-ghosts",
                        action="store_true", default=False,
                        help="Use malicious ghosts rather random ones.")
    parser.add_argument("-s", "--seed", help="Random seed.", type=np.uint32,
                        default=None)
    args = parser.parse_known_args()

    return args


if __name__ == "__main__":
    namespace_args = parse_args()
    args = namespace_args[0]
    if args.train:
        print("'Training' of the agent has begun.")
        if args.seed is not None:
            seeds = generate_deterministic_seeds(args.seed, args.num_games)
        else:
            seeds = None

        runner = LocalPacmanGameRunner(layout_path=args.layout,
                                       random_ghosts=not args.smart_ghosts,
                                       show_window=False,
                                       zoom_window=1.0,
                                       frame_time=0.1)
        games = []
        agent = PacQuiaoAgent(numTraining=int(0.75 * args.num_games))

        for i in tqdm.trange(args.num_games, leave=False):#range(args.num_games):
            if seeds is not None:
                random.seed(seeds[i])
            game = runner.run_game(agent)
            games.append(game)
        
        #write weights to file
        out_f = open("weights.txt", "w")
        values = []
        for key in agent.weights:
            values.append(agent.weights[key])
        formatted_str = ' '.join(str(val) for val in values)       
        out_f.write(formatted_str)
        out_f.close()
        
        scores = [game.state.getScore() for game in games]
        results = np.array([game.state.isWin() for game in games])
        print(agent.weights)
        print("Avg score:     {:0.2f}".format(np.mean(scores)))
        print("Best score:    {:0.2f}".format(max(scores)))
        print("Median score:  {:0.2f}".format(np.median(scores)))
        print("Worst score:   {:0.2f}".format(min(scores)))
        print("Win Rate:      {}/{} {:0.2f}".format(results.sum(), len(results), results.mean()))

    else:
        runner = StdIOPacmanRunner()
        games_num = int(input())
        # 'load' weights
        with open("weights.txt") as f:
            weights = [float(x) for x in f.readline().split()]
        #weights = [-122.94814697340031, 20.499589680796387, 37.03635695606525, 17.89614773437804]
        agent = PacQuiaoAgent(num_ghost_w=weights[0],
                              eats_food_w=weights[1],
                              scared_ghosts_w=weights[2],
                              closest_direction_w=weights[3],
                              numTraining=0,
                              alpha=0.0,
                              epsilon=0.0,
                              gamma=0.8)

        for _ in range(games_num):
            runner.run_game(agent)
