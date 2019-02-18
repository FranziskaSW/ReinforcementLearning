from policies import base_policy as bp
import numpy as np
import operator

EPSILON = 0.05

class Linear(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON

        return policy_args

    def init_run(self):
        self.r_sum = 0
        no_features = len(['FoodVicinity_1', 'FoodVicinity_2', 'FoodVicinity_3', 'FieldEmpty'])
        weights = np.random.uniform(0, 1, no_features)
        self.weights = weights / weights.sum()
        self.vicinity = vicinity_param
        self.features = np.zeros(no_features)

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                self.r_sum = 0
            else:
                self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

    def getVicinityMap(self, board, center):
        board_size = board.shape
        vicinity = self.vicinity

        r, c = center

        left = c - vicinity
        right = c + vicinity + 1
        top = r - vicinity
        bottom = r + vicinity + 1

        big_board = board

        if left < 0:
            left_patch = big_board[:, left]
            left = 0
            right = 2 * vicinity + 1
            big_board = np.hstack([left_patch, big_board])

        if right >= board_size[1]:
            right_patch = big_board[:, :(right % board_size[1] + 1)]
            big_board = np.hstack([big_board, right_patch])

        if top < 0:
            top_patch = big_board[top, :]
            top = 0
            bottom = 2 * vicinity + 1
            big_board = np.vstack([top_patch, big_board])

        if bottom >= board_size[0]:
            bottom_patch = big_board[:(bottom % board_size[0])]
            big_board = np.vstack([big_board, bottom_patch])

        return big_board[top:bottom, left:right]

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        board, head = new_state
        head_pos, direction = head
        vicinity = self.vicinity
        vicinity = 2
        weights = self.weights
        features = self.features

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        else:
            q_values = dict()
            for dir_idx, a in enumerate(list(np.random.permutation(bp.Policy.ACTIONS))):
                next_position = head_pos.move(bp.Policy.TURNS[direction][a])
                r = next_position[0]
                c = next_position[1]
                center_idx = (vicinity, vicinity)

                # TODO: that we also look through the walls

                VicinityMap = self.getVicinityMap(board, next_position)

                # look at the board in the relevant position:
                # check which food is in vicinity after move
                for feature_idx, food_value in enumerate([6, 7, 8]):

                    m = (VicinityMap == food_value)
                    food_positions = np.matrix(np.where(m)).T
                    print(feature_idx)

                    if food_positions.shape == (0, 2):  # if no food of this kind was found
                        dist = 10
                    else:
                        distances = []
                        for food_pos in food_positions:
                            print(food_pos)
                            x, y = food_pos.tolist()[0][0], food_pos.tolist()[0][1]
                            dist = abs(center_idx[0] - x) + abs(center_idx[1] - y)
                            distances.append(dist)
                            dist = min(distances)
                    features[feature_idx] = 10 - dist
                    # features[feature_idx] = 1/(dist + 0.001)

                # now check if next field is free
                if board[r, c] > 5 or board[r, c] < 0:
                    free = 5
                else:
                    free = 0
                features[3] = free
                # normalize features
                f = features / np.linalg.norm(features)

                q_value = f.dot(weights)  # + bias

                q_values.update({dir_idx: q_value})

            action_idx, q_value = max(q_values.items(), key=operator.itemgetter(1))
            return bp.Policy.ACTIONS[action_idx]