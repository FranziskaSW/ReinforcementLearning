from policies import base_policy as bp
import numpy as np
import operator

EPSILON = 0.05
GAMMA = 0.5
LEARNING_RATE = 0.05
VICINITY = 2
FEATURE_NUM = len(['Food1', 'Food2', 'Food3', 'FieldEmpty'])

class Linear(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        policy_args['learning_rate'] = float(policy_args['lr']) if 'lr' in policy_args else LEARNING_RATE
        policy_args['vicinity'] = float(policy_args['vicinity']) if 'vicinity' in policy_args else VICINITY

        return policy_args

    def init_run(self):
        self.r_sum = 0
        weights = np.random.uniform(0, 1, FEATURE_NUM)
        self.weights = weights / weights.sum()
        self.features = np.zeros(FEATURE_NUM)
        # self.last_actions = []  #
        self.last_qvalues = []
        self.last_deltas = []

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        last_features = self.last_features[-1]
        delta = self.last_deltas[-1]

        self.weights = self.weights - self.learning_rate * (delta * last_features).mean(axis=0)

        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(
                        self.r_sum), 'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                self.r_sum = 0
            else:
                self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

    def getQValue(self, field):
        # center = (self.vicinity, self.vicinity)
        weights = self.weights
        features = self.features

        # which food is there
        if field == 6: features[0] = 1
        if field == 7: field[1] = 1
        if field == 8: field[2] = 1
        # now check if next field is free
        if field < 0: field[3] = 1

        # normalize features
        f = features / np.linalg.norm(features) # is unnecessary because only one of the above can be true

        q_value = f.dot(weights)  # + bias
        return q_value, f


    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        board, head = new_state
        head_pos, direction = head

        if np.random.rand() < self.epsilon:
            action = np.random.choice(bp.Policy.ACTIONS)
            next_position = head_pos.move(bp.Policy.TURNS[direction][action])
            q_value, features = self.getQValue(board[next_position])

        else:
            res = {'features': [],
                   'q_values': np.zeros(3),
                   'action': []}
            for dir_idx, a in enumerate(list(np.random.permutation(bp.Policy.ACTIONS))):
                # get a Position object of the position in the relevant direction from the head:
                next_position = head_pos.move(bp.Policy.TURNS[direction][a])

                q_value, features_a = self.getQValue(board[next_position])  # getQValue(a, next_position, board, direction)
                res['features'].append(features_a)
                res['action'].append(a)
                res['q_values'][dir_idx] = q_value

            q_value, q_max_idx = np.max(res['q_values']), np.argmax(res['q_values'])
            features = res['features'][q_max_idx]
            action = res['action'][q_max_idx]

        delta = self.last_qvalues[-1] - (reward + (self.gamma * q_value))
        # self.last_actions = self.last_actions + [action]
        self.last_qvalues = self.last_qvalues + [q_value]
        self.last_features = self.last_features + [features]
        self.last_deltas = self.last_deltas + [delta]
        return action