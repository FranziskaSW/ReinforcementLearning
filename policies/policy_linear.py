from policies import base_policy as bp
import numpy as np
import operator
import pickle  # TODO remove after testing

EPSILON = 0.05
EPSILON_RATE = 0.9999
GAMMA = 0.8
LEARNING_RATE = 0.01
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

        return policy_args

    def init_run(self):
        self.r_sum = 0
        weights = np.matrix(np.random.uniform(0, 1, FEATURE_NUM))
        self.weights = weights / weights.sum()
        self.features = np.zeros(FEATURE_NUM)
        self.last_actions = []  #
        self.last_qvalues = []
        self.last_deltas = []
        self.last_features = []

    def put_stats(self):  # TODO remove after testing
        pickle.dump(self.loss, open(self.dir_name + '/last_game_loss.pkl', 'wb'))
        pickle.dump(self.test(), open(self.dir_name + '/last_test_loss.pkl', 'wb'))


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        #last_features = self.last_features[-1]
        #delta = self.last_deltas[-1]

        #last_features = self.last_features
        feature_mat = np.matrix(self.last_features)
        delta_mat = np.matrix(self.last_deltas)

        self.weights = self.weights - self.learning_rate * (delta_mat.dot(feature_mat))/len(self.last_deltas)

        self.last_features = []
        self.last_deltas = []

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
        weights = self.weights
        features = np.zeros(4)
        if field >= 0 and field < 6:
            return 0, features
        else:
            # which food is there
            if field == 6: idx=0
            if field == 7: idx=1
            if field == 8: idx=2
            # now check if next field is free
            if field < 0: idx=3
            features[idx] = 1
            q_value = weights[0, idx]  # + bias
            return q_value, features


    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        board, head = new_state
        head_pos, direction = head

        if np.random.rand() < self.epsilon:
            action = np.random.choice(bp.Policy.ACTIONS)
            next_position = head_pos.move(bp.Policy.TURNS[direction][action])
            r, c = next_position
            q_value, features = self.getQValue(board[r, c])

        else:
            res = {'features': [],
                   'q_values': np.zeros(3),
                   'action': []}
            for dir_idx, a in enumerate(list(np.random.permutation(bp.Policy.ACTIONS))):
                # get a Position object of the position in the relevant direction from the head:
                next_position = head_pos.move(bp.Policy.TURNS[direction][a])
                r, c = next_position

                q_value, features_a = self.getQValue(board[r, c])  # getQValue(a, next_position, board, direction)
                res['features'].append(features_a)
                res['action'].append(a)
                res['q_values'][dir_idx] = q_value

            q_value, q_max_idx = np.max(res['q_values']), np.argmax(res['q_values'])
            features = res['features'][q_max_idx]
            action = res['action'][q_max_idx]
        if round <= 1:
            delta = 0
        else:
            # print(self.last_qvalues, self.last_actions, self.last_deltas)
            delta = self.last_qvalues[-1] - (reward + (self.gamma * q_value))
        self.last_actions.append(action)
        self.last_qvalues.append(q_value)
        self.last_features.append(features)
        self.last_deltas.append(delta)
        self.epsilon = self.epsilon * EPSILON_RATE
        return action
