# 2_1_policy_value_iteration.py
import numpy as np
np.set_printoptions(precision=2)

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


# 퀴즈
# 다음 상태를 반환하는 함수를 만드세요
def get_next_state(state, action, size):
    row, col = state

    if action == LEFT and col > 0:
        col -= 1
    elif action == RIGHT and col < size-1:
        col += 1
    elif action == UP and row > 0:
        row -= 1
    elif action == DOWN and row < size-1:
        row += 1

    return row, col


def value_iteration(states, board, size):
    reward = -1
    grid = np.zeros_like(board)

    for s in states:
        values = []
        for action in [LEFT, DOWN, RIGHT, UP]:
            next_state = get_next_state(s, action, size)
            values.append(reward + board[next_state])

        grid[s] = np.max(values)

    return grid


def policy_iteration(states, board, size):
    reward = -1
    grid = np.zeros_like(board)

    for s in states:
        values = []
        for action in [LEFT, DOWN, RIGHT, UP]:
            next_state = get_next_state(s, action, size)
            values.append(reward + board[next_state])

        # grid[s] = np.sum([v * 0.25 for v in values])
        # grid[s] = np.sum([v / 4 for v in values])
        grid[s] = np.mean(values)

    return grid


def show_iteration(iter_func, loop, size):
    board = np.zeros([size, size])
    # print(board)

    states = [(i, j) for i in range(size) for j in range(size)]
    states.pop(0)
    states.pop(-1)
    # print(states)

    for i in range(loop):
        board = iter_func(states, board, size)
        # print(i+1, '\n', board)

    show_direction(states, board, size)


def show_direction(states, board, size):
    policy = ['    ']           # terminal state
    for s in states:
        rewards = []
        for action in [LEFT, DOWN, RIGHT, UP]:
            next_state = get_next_state(s, action, size)
            rewards.append(board[next_state])

        r_max = np.max(rewards)

        # r_max를 사용해서 방향을 arrows에 추가하세요(L, D, R, U)
        # 예시: ' DR ', 'L R '
        arrows = ''
        arrows += 'L' if r_max == rewards[LEFT] else ' '
        arrows += 'D' if r_max == rewards[DOWN] else ' '
        arrows += 'R' if r_max == rewards[RIGHT] else ' '
        arrows += 'U' if r_max == rewards[UP] else ' '

        policy.append(arrows)

    policy.append('    ')       # terminal state

    policy = np.reshape(policy, (-1, 4))
    print(policy)
    print(board)


show_iteration(value_iteration, 3, size=4)
show_iteration(policy_iteration, 159, size=4)

# sum(5) = 1 + 2 + 3 + 4 + 5
# sum(4) = 1 + 2 + 3 + 4
# sum(3) = 1 + 2 + 3
# sum(2) = 1 + 2
# sum(1) = 1

# sum(5) = sum(4) + 5
# sum(4) = sum(3) + 4

# SGD: Stochastic Gradient Descent
# 60000개 데이터
# 1. 60000개 모두 사용해서 1번 업데이트
# 2. 100개 사용해서 1번 업데이터(60000개에 대해 600번 업데이트)




