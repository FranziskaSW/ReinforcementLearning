import pickle
import time
import datetime
import multiprocessing as mp
import queue
import os
import argparse
import sys
import gzip
import subprocess as sp
import pandas as pd

import numpy as np
import scipy.signal as ss

from policies import base_policy
from policies import *

import os
import matplotlib.pyplot as plt

from copy import deepcopy

import time

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.style.use('ggplot')

TEST_POLICY = 'Linear'

EMPTY_VAL = -1
MAX_PLAYERS = 5
OBSTACLE_VAL = 5
REGULAR_RENDER_MAP = {EMPTY_VAL: ' ', OBSTACLE_VAL: '+'}
FOOD_RENDER_MAP = {6: '*', 7: '$', 8: 'X'}
FOOD_VALUE_MAP = {6: 1, 7: 3, 8: 0}
FOOD_REWARD_MAP = {6: 2, 7: 5, 8: -1}
THE_DEATH_PENALTY = -5

ILLEGAL_MOVE = "Illegal Action: the default action was selected instead. Player tried action: "
NO_RESPONSE = "No Response: player took too long to respond with action. This is No Response #"
PLAYER_INIT_TIME = 60
UNRESPONSIVE_PLAYER = "Unresponsive Player: the player hasn't responded in too long... SOMETHING IS WRONG!!"

STATUS_SKIP = 100
TOO_SLOW_THRESHOLD = 3
UNRESPONSIVE_THRESHOLD = 50
LEARNING_TIME = 5


def clear_q(q):
    """
    given a queue, empty it.
    """
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def days_hours_minutes_seconds(td):
    """
    parse time for logging.
    """
    return td.days, td.seconds // 3600, (td.seconds // 60) % 60, td.seconds % 60


def random_partition(num, max_part_size):
    parts = []
    while num > 0:
        parts.append(np.random.randint(1, min(max_part_size, num + 1)))
        num -= parts[-1]
    return parts


class Position():
    def __init__(self, position, board_size):
        self.pos = position
        self.board_size = board_size

    def __getitem__(self, key):
        return self.pos[key]

    def __add__(self, other):
        return Position(((self[0] + other[0]) % self.board_size[0],
                         (self[1] + other[1]) % self.board_size[1]),
                        self.board_size)

    def move(self, dir):
        if dir == 'E': return self + (0, 1)
        if dir == 'W': return self + (0, -1)
        if dir == 'N': return self + (-1, 0)
        if dir == 'S': return self + (1, 0)
        raise ValueError('unrecognized direction')


class Agent(object):
    SHUTDOWN_TIMEOUT = 60  # seconds until policy is considered unresponsive

    def __init__(self, id, policy, policy_args, board_size, logq, game_duration, score_scope, dir_name):
        """
        Construct a new player
        :param id: the player id (the value of the player positions in the board
        :param policy: the class of the policy to be used by the player
        :param policy_args: string (name, value) pairs that the policy can parse to arguments
        :param board_size: the size of the game board (height, width)
        :param logq: a queue for message logging through the game
        :param game_duration: the expected duration of the game in turns
        :param score_scope: the amount of rounds at the end of the game which count towards the score
        """

        self.id = id
        self.len = 0
        self.policy_class = policy
        self.round = 0
        self.unresponsive_count = 0
        self.too_slow = False

        self.sq = mp.Queue()
        self.aq = mp.Queue()
        self.mq = mp.Queue()

        self.logq = logq
        self.policy = policy(policy_args, board_size, self.sq, self.aq, self.mq, logq, id,
                             game_duration, score_scope, dir_name)
        self.policy.daemon = True
        self.policy.start()

    def handle_state(self, round, prev_state, prev_action, reward, new_state):
        """
        given the new state and previous state-action-reward, pass the information
        to the policy for action selection and/or learning.
        """

        self.round = round
        clear_q(self.sq)  # remove previous states from queue if they weren't handled yet
        self.sq.put((round, prev_state, prev_action, reward, new_state, self.too_slow))

    # get statistics from agent TODO REMOVE AFTER TESTING
    def get_stats(self):
        pass
        # stats = self.mq.get_nowait()
        # clear_q(self.mq)
        # return stats

    def get_action(self):
        """
        get waiting action from the policy's action queue. if there is no action
        in the queue, pick 'F' and log the unresponsiveness error.
        :return: action from {'R','L','F'}.
        """
        try:
            round, action = self.aq.get_nowait()
            if round != self.round:
                raise queue.Empty()
            elif action not in base_policy.Policy.ACTIONS:
                self.logq.put((str(self.id), "ERROR", ILLEGAL_MOVE + str(action)))
                raise queue.Empty()
            else:
                self.too_slow = False
                self.unresponsive_count = 0

        except queue.Empty:
            self.unresponsive_count += 1
            action = base_policy.Policy.DEFAULT_ACTION
            if self.unresponsive_count <= UNRESPONSIVE_THRESHOLD:
                self.logq.put((str(self.id), "ERROR",
                               NO_RESPONSE + str(self.unresponsive_count) + " in a row!"))
            else:
                self.logq.put((str(self.id), "ERROR", UNRESPONSIVE_PLAYER))
                self.unresponsive_count = TOO_SLOW_THRESHOLD
            if self.unresponsive_count > TOO_SLOW_THRESHOLD:
                self.too_slow = True

        clear_q(self.aq)  # clear the queue from unhandled actions
        return action

    def shutdown(self):
        """
        shutdown the agent in the end of the game. the function asks the agent
        to save it's model and returns the saved model, which needs to be a data
        structure that can be pickled.
        :return: the model data structure.
        """

        clear_q(self.sq)
        clear_q(self.aq)
        self.sq.put(None)  # shutdown signal
        self.policy.join()
        return


class Game(object):
    @staticmethod
    def log(q, file_name, on_screen=True):
        start_time = datetime.datetime.now()
        logfile = None
        if file_name:
            logfile = gzip.GzipFile(file_name,
                                    'w') if file_name.endswith(
                '.gz') else open(file_name, 'wb')
        for frm, type, msg in iter(q.get, None):
            td = datetime.datetime.now() - start_time
            msg = '%i::%i:%i:%i\t%s\t%s\t%s' % (
                days_hours_minutes_seconds(td) + (frm, type, msg))
            if logfile: logfile.write((msg + '\n').encode('ascii'))
            if on_screen: print(msg)
        if logfile: logfile.close()

    def _find_empty_slot(self, shape=(1, 3)):
        is_empty = np.asarray(self.board == EMPTY_VAL, dtype=int)
        match = ss.convolve2d(is_empty, np.ones(shape), mode='same') == np.prod(shape)
        if not np.any(match): raise ValueError('no empty slots of requested shape')
        r = np.random.choice(np.nonzero(np.any(match, axis=1))[0])
        c = np.random.choice(np.nonzero(match[r, :])[0])
        return Position((r, c), self.board_size)

    def __init__(self, args):
        self.__dict__.update(args.__dict__)

        # check that the number of players is OK:
        self.n = len(self.policies)
        assert self.n <= MAX_PLAYERS, "Too Many Players!"

        self.round = 0

        # check for playback:
        self.is_playback = False
        if self.playback_from is not None:
            self.archive = open(self.playback_from, 'rb')
            dict = pickle.load(self.archive)
            self.__dict__.update(dict)
            self.is_playback = True
            self.record_to = None
            self.record = False
            self.to_render = args.to_render
            return

        # init logger
        self.logq = mp.Queue()
        to_screen = not self.to_render
        self.logger = mp.Process(target=self.log, args=(self.logq, self.log_file, to_screen))
        self.logger.start()

        # initialize the board:
        self.item_count = 0
        self.board = EMPTY_VAL * np.ones(self.board_size, dtype=int)
        self.previous_board = None

        # initialize obstacles:
        for size in random_partition(int(self.obstacle_density * np.prod(self.board_size)),
                                     np.min(self.board_size)):
            if np.random.rand(1) > 0.5:
                pos = self._find_empty_slot(shape=(size, 1))
                for i in range(size):
                    new_pos = pos + (i, 0)
                    self.board[new_pos[0], new_pos[1]] = OBSTACLE_VAL
            else:
                pos = self._find_empty_slot(shape=(1, size))
                for i in range(size):
                    new_pos = pos + (i, 0)
                    self.board[new_pos[0], new_pos[1]] = OBSTACLE_VAL
            self.item_count += size

        # initialize players:
        self.rewards, self.players, self.scores, self.directions, self.actions, self.growing, self.size, self.chains, self.previous_heads = \
            [], [], [], [], [], [], [], [], []
        for i, (policy, pargs) in enumerate(self.policies):
            self.rewards.append(0)
            self.actions.append(None)
            self.previous_heads.append(None)
            self.scores.append([0])
            self.players.append(
                Agent(i, policy, pargs, self.board_size, self.logq, self.game_duration,
                      self.score_scope, self.dir_name))
            chain, player_size, growing, direction = self.init_player()
            self.chains.append(chain)
            self.size.append(player_size)
            self.growing.append(growing)
            self.directions.append((direction))
            for position in chain:
                self.board[position[0], position[1]] = i

        # configure symbols for rendering
        self.render_map = {p.id: chr(ord('1') + p.id) for p in self.players}
        self.render_map.update(REGULAR_RENDER_MAP)
        self.render_map.update(FOOD_RENDER_MAP)

        # finally, if it's a recording, then record
        if self.record_to is not None and self.playback_from is None:
            self.archive = open(self.record_to, 'wb')
            dict = self.__dict__.copy()
            del dict['players']  # remove problematic objects that are irrelevant to playback.
            del dict['archive']
            del dict['logq']
            del dict['logger']
            del dict['playback_initial_round']
            del dict['playback_final_round']
            pickle.dump(dict, self.archive)
        self.record = self.record_to is not None

        # wait for player initialization (Keras loading time):
        time.sleep(self.player_init_time)

    def init_player(self):

        # initialize the position and direction of the player:
        dir = np.random.choice(list(base_policy.Policy.TURNS.keys()))
        shape = (1, 3) if dir in ['W', 'E'] else (3, 1)
        pos = self._find_empty_slot(shape)

        # gather stats about the player:
        chain = []
        chain.append(pos)
        chain.append(pos.move(dir))
        player_size = 2
        growing = self.init_player_size - 2

        return chain, player_size, growing, dir

    def reset_player(self, id):

        positions = np.array(np.where(self.board == id))
        for pos in range(positions.shape[1]):
            self.board[positions[0, pos], positions[1, pos]] = EMPTY_VAL

        # turn parts of the corpse into food:
        food_n = np.random.binomial(positions.shape[1], self.food_ratio)
        if self.item_count + food_n < self.max_item_density * np.prod(self.board_size):
            subidx = np.array(np.random.choice(positions.shape[1], size=food_n, replace=False))
            if len(subidx) > 0:
                randfood = np.random.choice(list(FOOD_VALUE_MAP.keys()), food_n)
                for i, idx in enumerate(subidx):
                    self.board[positions[0, idx], positions[1, idx]] = randfood[i]
                self.item_count += food_n
        return self.init_player()

    def randomize(self):
        if np.random.rand(1) < self.random_food_prob:
            if self.item_count < self.max_item_density * np.prod(self.board_size):
                randfood = np.random.choice(list(FOOD_VALUE_MAP.keys()), 1)
                slot = self._find_empty_slot((1, 1))
                self.board[slot[0], slot[1]] = randfood
                self.item_count += 1

    def move_snake(self, id, action):

        # delete the tail if the snake isn't growing:
        if self.growing[id] > 0:
            self.growing[id] -= 1
            self.size[id] += 1
        else:
            self.board[self.chains[id][0][0], self.chains[id][0][1]] = EMPTY_VAL
            del self.chains[id][0]

        # move the head:
        if action != 'F':  # turn in the relevant direction
            self.directions[id] = base_policy.Policy.TURNS[self.directions[id]][action]
        self.chains[id].append(self.chains[id][-1].move(self.directions[id]))
        self.board[self.chains[id][-1][0], self.chains[id][-1][1]] = id

    def play_a_round(self):

        # randomize the players:
        pperm = np.random.permutation([(i, p) for i, p in enumerate(self.players)])

        # distribute states and rewards on previous round
        for i, p in pperm:
            current_head = (self.chains[p.id][-1], self.directions[p.id])
            if self.previous_board is None:
                p.handle_state(self.round, None, self.actions[p.id], self.rewards[p.id],
                               (self.board, current_head))
            else:
                p.handle_state(self.round, (self.previous_board, self.previous_heads[p.id]),
                               self.actions[p.id], self.rewards[p.id], (self.board, current_head))
            self.previous_heads[p.id] = current_head
        self.previous_board = np.copy(self.board)

        # wait and collect actions
        time.sleep(self.policy_action_time)
        actions = {p: p.get_action() for _, p in pperm}
        if self.round % LEARNING_TIME == 0 and self.round > 5:
            time.sleep(self.policy_learn_time)

        # get the interactions of the players with the board:
        for _, p in pperm:
            action = actions[p]
            self.actions[p.id] = action
            move_to = self.chains[p.id][-1].move(
                base_policy.Policy.TURNS[self.directions[p.id]][action])

            # reset the player if he died:
            if self.board[move_to[0], move_to[1]] != EMPTY_VAL and self.board[
                move_to[0], move_to[1]] not in FOOD_VALUE_MAP:
                chain, player_size, growing, dir = self.reset_player(p.id)
                self.chains[p.id] = chain
                self.size[p.id] = player_size
                self.growing[p.id] = growing
                self.directions[p.id] = dir
                self.rewards[p.id] = THE_DEATH_PENALTY
                self.scores[p.id].append(self.rewards[p.id])
                for pos in chain:
                    self.board[pos[0], pos[1]] = p.id

            # otherwise, move the player on the board:
            else:
                self.rewards[p.id] = 0
                if self.board[move_to[0], move_to[1]] in FOOD_VALUE_MAP.keys():
                    self.rewards[p.id] += FOOD_REWARD_MAP[self.board[move_to[0], move_to[1]]]
                    self.growing[p.id] += FOOD_VALUE_MAP[
                        self.board[move_to[0], move_to[1]]]  # start growing
                    self.item_count -= 1

                self.move_snake(p.id, action)
                self.scores[p.id].append(self.rewards[p.id])

        # update the food on the board:
        self.randomize()
        self.round += 1

    def render(self, r):

        if os.name == 'nt':
            os.system('cls')  # clear screen for Windows
        else:
            print(chr(27) + "[2J")  # clear screen for linux

        # print the scores:
        print("Time Step: " + str(r) + "/" + str(self.game_duration))
        for i in range(len(self.scores)):
            scope = np.min([len(self.scores[i]), self.score_scope])
            print("Player " + str(i + 1) + ": " + str(
                "{0:.4f}".format(np.mean(self.scores[i][-scope:]))))

        # print the board:
        horzline = '-' * (self.board.shape[1] + 2)
        board = [horzline]
        for r in range(self.board.shape[0]):
            board.append('|' + ''.join(
                self.render_map[self.board[r, c]] for c in range(self.board.shape[1])) + '|')
        board.append(horzline)
        print('\n'.join(board))

    def run(self):
        try:
            r = 0
            while r < self.game_duration:
                r += 1
                if self.to_render:
                    if not (self.is_playback and (
                                    r < self.playback_initial_round or r > self.playback_final_round)):
                        self.render(r)
                        time.sleep(self.render_rate)
                else:
                    if r % STATUS_SKIP == 0:
                        print("At Round " + str(r) + " the scores are:")
                        for i, s in enumerate(self.scores):
                            scope = np.min([len(self.scores[i]), self.score_scope])
                            print("Player " + str(i + 1) + ": " + str(
                                str("{0:.4f}".format(np.mean(self.scores[i][-scope:])))))
                if self.is_playback:
                    try:
                        idx, vals, self.scores = pickle.load(self.archive)
                        self.board[idx] = vals
                    except EOFError:
                        break
                    if r > self.playback_final_round:
                        break

                else:
                    if self.record:
                        prev = self.board.copy()
                    self.play_a_round()
                    if self.record:
                        idx = np.nonzero(self.board - prev != 0)
                        pickle.dump((idx, self.board[idx], self.scores), self.archive)
        finally:
            if self.record or self.is_playback:
                self.archive.close()

            if not self.is_playback:
                output = [','.join(['game_id', 'player_id', 'policy', 'score'])]
                game_id = str(abs(id(self)))
                for p, s in zip(self.players, self.scores):
                    p.shutdown()
                    pstr = str(p.policy).split('<')[1].split('(')[0]
                    scope = np.min([len(s), self.score_scope])
                    p_score = np.mean(s[-scope:])
                    oi = [game_id, str(p.id), pstr, str("{0:.4f}".format(p_score))]
                    output.append(','.join(oi))

                with open(self.output_file, 'w') as outfile:
                    outfile.write('\n'.join(output))
                self.logq.put(None)
                self.logger.join()


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_argument_group('I/O')
    g.add_argument('--record_to', '-rt', type=str, default=None,
                   help="file path to which game will be recorded.")
    g.add_argument('--playback_from', '-p', type=str, default=None,
                   help='file path from which game will be played-back (overrides record_to)')
    g.add_argument('--playback_initial_round', '-pir', type=int, default=0,
                   help='round in which to start the playback')
    g.add_argument('--playback_final_round', '-pfr', type=int, default=1000,
                   help='round in which to end the playback')
    g.add_argument('--log_file', '-l', type=str, default=None,
                   help="a path to which game events are logged. default: game.log")
    g.add_argument('--output_file', '-o', type=str, default=None,
                   help="a path to a file in which game results are written. default: game.out")
    g.add_argument('--to_render', '-r', type=int, default=0,
                   help="whether game should not be rendered")
    g.add_argument('--render_rate', '-rr', type=float, default=0.1,
                   help='frames per second, note that the policy_wait_time bounds on the rate')
    g.add_argument('--sessions', '-ss', type=int, default=1,
                   help='number of sessions, for testing')
    g.add_argument('--test_param', '-tp', type=str, default='epsilon',
                   help='parameter to test')
    g.add_argument('--dir_suffix', '-ds', type=str, default='',
                   help='test results directory suffix')
    g.add_argument('--dir_name', '-dn', type=str, default='',
                   help='test results directory name')

    g = p.add_argument_group('Game')
    g.add_argument('--board_size', '-bs', type=str, default='(20,60)',
                   help='a tuple of (height, width)')
    g.add_argument('--obstacle_density', '-od', type=float, default=.04,
                   help='the density of obstacles on the board')
    g.add_argument('--policy_wait_time', '-pwt', type=float, default=0.01,
                   help='seconds to wait for policies to respond with actions')
    g.add_argument('--random_food_prob', '-fp', type=float, default=.2,
                   help='probability of a random food appearing in a round')
    g.add_argument('--max_item_density', '-mid', type=float, default=.25,
                   help='maximum item density in the board (not including the players)')
    g.add_argument('--food_ratio', '-fr', type=float, default=.2,
                   help='the ratio between a corpse and the number of food items it produces')
    g.add_argument('--game_duration', '-D', type=int, default=10000,
                   help='number of rounds in the session')
    g.add_argument('--policy_action_time', '-pat', type=float, default=0.01,
                   help='seconds to wait for agents to respond with actions')
    g.add_argument('--policy_learn_time', '-plt', type=float, default=0.1,
                   help='seconds to wait for agents to improve policy')
    g.add_argument('--player_init_time', '-pit', type=float, default=PLAYER_INIT_TIME,
                   help='seconds to wait for agents to initialize in the beginning of the session')

    g = p.add_argument_group('Players')
    g.add_argument('--score_scope', '-s', type=int, default=1000,
                   help='The score is the average reward during the last score_scope rounds of the session')
    g.add_argument('--init_player_size', '-is', type=int, default=5,
                   help='player length at start, minimum is 3')
    g.add_argument('--policies', '-P', type=str, default=None,
                   help='a string describing the policies to be used in the game, of the form: '
                        '<policy_name>(<arg=val>,*);+.\n'
                        'e.g. MyPolicy(layer1=100,layer2=20);YourPolicy(your_params=123)')

    args = p.parse_args()

    # set defaults
    code_path = os.path.split(os.path.abspath(__file__))[0] + os.path.sep
    if not args.record_to:
        args.__dict__['record_to'] = None
    if args.log_file is None:
        args.__dict__['log_file'] = code_path + 'game.log'
    if args.output_file is None:
        args.__dict__['output_file'] = code_path + 'game.out'
    if args.playback_from is not None:
        args.__dict__['record_to'] = None
        args.__dict__['output_file'] = None
        args.__dict__['log_file'] = None

    args.__dict__['board_size'] = [int(x) for x in args.board_size[1:-1].split(',')]
    plcs = []
    if args.policies is not None: plcs.extend(args.policies.split(';'))
    args.__dict__['policies'] = [base_policy.build(p) for p in plcs]

    return args


def moving_average(a, n=20):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_single_loss(loss, param, test_param, dir_name, filename_suffix=''):
    plt.plot(loss, label=test_param + '=' + str(param))
    plt.xlabel('Round')
    plt.ylabel('MSE')
    plt.title('Loss')
    plt.legend(loc=0)
    plt.savefig(dir_name + '/single_losses/loss_' + test_param + '=' + str(param) + filename_suffix + '.png')
    plt.clf()

    # also plot averaged loss:
    plt.plot(moving_average(loss, n=100), label=test_param + '=' + str(param))
    plt.xlabel('Round')
    plt.ylabel('MSE')
    plt.title('Loss')
    plt.legend(loc=0)
    plt.savefig(dir_name + '/single_losses/moving_avg_loss_' + test_param + '=' + str(param) + filename_suffix + '.png')
    plt.clf()


def plot_averaged_scores(scores, score_scope, params, test_param, dir_name):
    for i in range(len(scores)):
        v = params[test_param][i]
        plt.plot(moving_average(scores[i], n=score_scope), label=test_param + '=' + str(v))
    plt.xlabel('Round')
    plt.ylabel('Score (Averaged by Scope)')
    plt.title('Score')
    plt.legend(loc=0)
    plt.savefig(dir_name + '/scores_' + test_param + '.png')
    plt.clf()


def plot_game_loss(game_loss, params, test_param, dir_name):
    for i in range(len(params[test_param])):
        v = params[test_param][i]
        plt.plot(game_loss[i], label=test_param + '=' + str(v))
        # plt.plot(x, moving_average(stats[i][stat_name]), label=test_param + '=' + str(v))
    plt.xlabel('Round')
    plt.ylabel('MSE')
    plt.title('Loss')
    plt.legend(loc=0)
    plt.savefig(dir_name + '/loss_' + test_param + '.png')
    plt.clf()

    # also plot averaged game loss:
    for i in range(len(params[test_param])):
        v = params[test_param][i]
        plt.plot(moving_average(game_loss[i], n=100), label=test_param + '=' + str(v))
        # plt.plot(x, moving_average(stats[i][stat_name]), label=test_param + '=' + str(v))
    plt.xlabel('Round')
    plt.ylabel('MSE')
    plt.title('Loss')
    plt.legend(loc=0)
    plt.savefig(dir_name + '/moving_avg_loss_' + test_param + '.png')
    plt.clf()


def plot_final_loss(final_loss, params, test_param, dir_name):
    plt.plot(params[test_param], final_loss)
    plt.xlabel(param2str.get(test_param, test_param).title())
    plt.ylabel('MSE')
    plt.title('Loss by ' + param2str.get(test_param, test_param).title())
    plt.savefig(dir_name + '/final_loss_' + test_param + '.png')
    plt.clf()


def get_stats(dir_name):
    return pickle.load(open(dir_name + '/last_game_loss.pkl', 'rb'))


def get_test_loss(dir_name):
    return pickle.load(open(dir_name + '/last_test_loss.pkl', 'rb'))


def search_1d_param_space(test_param, dir_suffix, sessions, num_players):
    global players, args, params
    dir_name = test_param + dir_suffix
    stats = []
    scores = []
    final_loss = []
    dirs = [dir_name, dir_name + '/logs', dir_name + '/pickles', dir_name + '/single_losses']
    for dir in dirs:
        if not os.path.isdir(dir):
            os.mkdir(dir)
    players.append('')  # place holder for test player
    num_params = len(params[test_param])
    other_players_scores = np.zeros((num_params, num_players + 2, sessions))
    for i, v in enumerate(params[test_param]):
        print('Starting with ' + param2str.get(test_param, test_param).title() + '=' + str(v) +
              ' (' + str(i+1) + '/' + str(len(params[test_param])) + ')')
        stats.append([])
        scores.append([])
        final_loss.append([])
        min_scores_length = min_stats_length = np.inf

        for session in range(sessions):
            print('Session ' + str(session+1) + '/' + str(sessions))
            game_name = test_param + '=' + str(v) + '_session=' + str(session)  # .split('.')[-1]
            args.__dict__['log_file'] = dir_name + '/logs/' + game_name + '.log'
            args.__dict__['output_file'] = dir_name + '/logs/' + game_name + '.out'

            players[-1] = TEST_POLICY + '(' + test_param + '=' + str(v) + ')'
            args.__dict__['policies'] = [base_policy.build(p) for p in players]
            args.__dict__['dir_name'] = dir_name

            g = Game(args)
            g.run()
            time.sleep(2)
            stats[-1].append(get_stats(dir_name))
            final_loss[-1].append(get_test_loss(dir_name))
            scores[-1].append(g.scores[-1])
            min_scores_length = min(len(scores[-1][-1]), min_scores_length)
            min_stats_length = min(len(stats[-1][-1]), min_stats_length)

            for k in range(len(g.scores)):
                other_players_scores[i][k][session] = np.mean(g.scores[k][-args.score_scope:])

            plot_single_loss(stats[-1][-1], v, test_param, dir_name, filename_suffix='_session=' + str(session))
        other_players_scores[i][-2] = other_players_scores[i].max(axis=0)
        other_players_scores[i][-1] = other_players_scores[i][-2] > other_players_scores[i][-3]

        if sessions > 1:
            for k in range(sessions):
                stats[-1][k] = stats[-1][k][:min_stats_length]
                scores[-1][k] = scores[-1][k][:min_scores_length]

            stats[-1] = np.mean(stats[-1], axis=0)
            final_loss[-1] = np.mean(final_loss[-1])
            scores[-1] = np.mean(scores[-1], axis=0)
        else:
            stats[-1] = stats[-1][-1]
            final_loss[-1] = final_loss[-1][-1]
            scores[-1] = scores[-1][-1]
        plot_single_loss(stats[-1], v, test_param, dir_name)

    # showcase learning
    plot_game_loss(stats, params=params, test_param=test_param, dir_name=dir_name)
    plot_final_loss(final_loss, params=params, test_param=test_param, dir_name=dir_name)
    plot_averaged_scores(scores, args.score_scope, params, test_param, dir_name)

    score_index = []
    for v in params[test_param]:
        for i in range(1, num_players+1):
            score_index.append(test_param + '=' + str(v) + ', player ' + str(i))
        score_index.append('other players max')
        score_index.append('higher than yours')
    other_players_scores = other_players_scores.reshape((num_params*(num_players + 2), sessions))
    other_players_scores = pd.DataFrame(other_players_scores,
                                        index=score_index,
                                        columns=['session ' + str(i) for i in range(1, sessions+1)])
    other_players_scores.to_excel(dir_name + '/pickles/final_scores.xlsx')

    pickle.dump(stats, open(dir_name + '/pickles/stats_n_sessions=' + str(sessions) + '.pkl', 'wb'))
    pickle.dump(final_loss, open(dir_name + '/pickles/final_loss.pkl', 'wb'))
    pickle.dump(scores, open(dir_name + '/pickles/scores.pkl', 'wb'))
    pickle.dump((test_param, params[test_param]), open(dir_name + '/pickles/parameters.pkl', 'wb'))


# def search_2d_param_space(test_params):
#     global players, args, params
#     stats = []
#
#     test_params = test_params.sort()
#     a = test_params[0]
#     b = test_params[1]
#     dir_suffix = a + '_x_' + b
#     if not os.path.isdir(dir_suffix):
#         os.mkdir(dir_suffix)
#
#     for param_a in params[a]:
#         for param_b in params[b]:
#             game_name = a + '=' + str(param_a) + '_' + b + '=' + str(param_b)
#             args.__dict__['log_file'] = dir_suffix + '/' + game_name + '.log'
#             args.__dict__['output_file'] = dir_suffix + '/' + game_name + '.out'
#             players[-1] = 'FeatDeep(' + a + '=' + str(param_a) + ',' + b + str(param_b) + ')'
#             args.__dict__['policies'] = [base_policy.build(p) for p in players]
#
#             g = Game(args)
#             g.run()
#             plt.scatter(range(len(g.scores[-1])), g.scores[-1], label=players[-1])
#             plt.title('scores')
#             plt.legend(loc=0)
#             plt.savefig(dir_suffix + '/scores_' + game_name + '.png')
#             plt.clf()
#
#             stats.append(g.players[-1].get_stats())


param2str = {'bs': 'batch size', 'lr': 'learning rate', 'epsilon': 'epsilon', 'gamma': 'gamma'}

if __name__ == '__main__':
    args = parse_args()

    AVOIDS = 4
    players = []
    if AVOIDS > 1:
        # for i in range(AVOIDS):
        #     players.append('Avoid(epsilon=0.0)')
        for eps in np.linspace(0, 0.15, AVOIDS):
            players.append('Avoid(epsilon=' + str(eps) + ')')
    else:
        players.append('Avoid(epsilon=0.1)')

    # create our agent:
    params = {'epsilon': [0.15, 0.2, 0.3, 0.4],
              'gamma': [0.1, 0.3, 0.5, 0.7, 0.9],
              'learning_rate': [0.01, 0.001, 0.0001, 0.00001],
              'batch_size': [4, 8, 16, 32, 64]
              }

    # test_params = ['epsilon', 'gamma', 'lr', 'bs']
    test_params = args.__dict__['test_param'].split() # tp
    sessions = args.__dict__['sessions'] # ss
    dir_suffix = args.__dict__['dir_suffix'] # ds

    for param in test_params:
        search_1d_param_space(param, dir_suffix, sessions, num_players=AVOIDS+1)
