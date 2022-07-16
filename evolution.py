import copy
from matplotlib.pyplot import step
import numpy as np
from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        # final_players = self.top_k_algorithm(players, num_players)

        # TODO (Additional: Implement roulette wheel here)
        # final_players = self.roulette_wheel(players, num_players)
        
        # TODO (Additional: Implement SUS here)
        # final_players = self.stochastic_universal_sampling(players, num_players)

        # TODO (Additional: Implement Q-tournament)
        final_players = self.tournament_selection(players, num_players, 20)
        # TODO (Additional: Learning curve)
        # print(players)
        self.update_generation_information(players)

        return final_players

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )
            new_players = []
            selected_parents = self.tournament_selection(prev_players, 2*num_players, 5)
            # selected_indices = np.random.choice(len(prev_players), 2*num_players)
            # selected_parents = [prev_players[i] for i in selected_indices]
            selected_pairs = list(zip(selected_parents[::2], selected_parents[1::2]))
            for pair in selected_pairs:
                offspring = self.crossover(*pair)
                mutated_offspring = self.mutate(offspring)
                new_players.append(mutated_offspring)

            assert len(new_players) == num_players
            return new_players

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    def top_k_algorithm(self, players, num_players):

        sorted_players = sorted(players, key=lambda x: x.fitness, reverse=True)
        final_players = sorted_players[:num_players]
        return final_players

    def roulette_wheel(self, players, num_players):

        max_fitness = sum([player.fitness for player in players])
        players_probs = [player.fitness/max_fitness for player in players]
        selected_indices = np.random.choice(len(players), num_players, p=players_probs)
        final_players = [players[i] for i in selected_indices]
        return final_players

    def stochastic_universal_sampling(self, players, num_players):

        final_players = []
        max_fitness = sum([player.fitness for player in players])
        players_probs = [player.fitness/max_fitness for player in players]

        step_size = 1.0/num_players
        r = np.random.uniform(0, step_size)

        cum_players_probs = 0
        for i in range(len(players)):
            player_prob = players_probs[i]
            cum_players_probs += player_prob
            if r <= cum_players_probs:
                final_players.append(players[i])
                r += step_size

        if len(final_players) < num_players:
            final_players.append(players[-1])
        return final_players

    def tournament_selection(self, players, num_players, k):
        
        final_players = []
        for _ in range(num_players):
            selected_indices = np.random.choice(len(players), k)
            selected_players = [players[i] for i in selected_indices]

            selected_players.sort(key=lambda x: x.fitness, reverse=True)
            final_players.append(selected_players[0])

        return final_players

    def update_generation_information(self, players, file_path='generation_information.gi'):
        # print(type(players))
        players_fitness = [player.fitness for player in players]
        min_fitness = min(players_fitness)
        max_fitness = max(players_fitness)
        avg_fitness = sum(players_fitness) / len(players_fitness)
        print(f'min: {min_fitness}, avg: {avg_fitness}, max: {max_fitness}')
        with open(file_path, 'a') as f:
            f.write(repr(players_fitness)+'\n')
            f.flush()

    def crossover(self, player1 :Player, player2 :Player):
        
        assert player1.nn.layer_num == player2.nn.layer_num
        layer_num = player1.nn.layer_num
        new_player = self.clone_player(player1)
        for i in range(layer_num):
            W_shape = new_player.nn.parameters[f'W_{i+1}'].shape
            b_shape = new_player.nn.parameters[f'b_{i+1}'].shape

            # new_player.nn.parameters[f'W_{i+1}'] =  new_player.nn.parameters[f'W_{i+1}'].ravel()
            # new_player.nn.parameters[f'b_{i+1}'] =  new_player.nn.parameters[f'b_{i+1}'].ravel()

            player1_W = player1.nn.parameters[f'W_{i+1}']
            player1_b = player1.nn.parameters[f'b_{i+1}'].ravel()

            player2_W = player2.nn.parameters[f'W_{i+1}']
            player2_b = player2.nn.parameters[f'b_{i+1}'].ravel()

            for j in range(W_shape[1]):
                if np.random.uniform(0, 1) > 0:
                    if j % 4 < 2:
                        new_player.nn.parameters[f'W_{i+1}'][:, j] = player1_W[:, j]
                    else:
                    #     if np.random.uniform(0, 1) > 0.5:
                        new_player.nn.parameters[f'W_{i+1}'][:, j] = player2_W[:, j]
                    #     else:
                    #         new_player.nn.parameters[f'W_{i+1}'][j] = player2_W[j]

            for j in range(len(player1_b)):
                if np.random.uniform(0, 1) > 0:
                    if (j % b_shape[1] % 4 < 2):
                        # new_player.nn.parameters[f'b_{i+1}'][j] = (player1_b[j] + player2_b[j]) / 2
                        new_player.nn.parameters[f'b_{i+1}'][j] = player1_b[j]
                    else:
                    #     if np.random.uniform(0, 1) > 0.5:
                    #         new_player.nn.parameters[f'b_{i+1}'][j] = player1_b[j]
                    #     else:
                        new_player.nn.parameters[f'b_{i+1}'][j] = player2_b[j]

            new_player.nn.parameters[f'W_{i+1}'] =  new_player.nn.parameters[f'W_{i+1}'].reshape(*W_shape)
            new_player.nn.parameters[f'b_{i+1}'] =  new_player.nn.parameters[f'b_{i+1}'].reshape(*b_shape)

        return new_player

    def mutate(self, player :Player):
        
        layer_num = player.nn.layer_num
        for i in range(layer_num):
            W_shape = player.nn.parameters[f'W_{i+1}'].shape
            b_shape = player.nn.parameters[f'b_{i+1}'].shape

            max_W = np.max(player.nn.parameters[f'W_{i+1}'])
            min_W = np.min(player.nn.parameters[f'W_{i+1}'])

            max_b = np.max(player.nn.parameters[f'b_{i+1}'])
            min_b = np.min(player.nn.parameters[f'b_{i+1}'])

            if np.random.uniform(0, 1) > 0.5:
                player.nn.parameters[f'W_{i+1}'] += np.random.normal(size=W_shape)

            if np.random.uniform(0, 1) > 0.5:
                player.nn.parameters[f'b_{i+1}'] += np.random.normal(size=b_shape)

        return player
        