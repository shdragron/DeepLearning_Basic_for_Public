# 1_4_frozen_lake_game.py
import readchar
import gym

# 퀴즈
# x를 입력할 때까지 반복 입력받는 코드를 만드세요
# while True:
#     c = readchar.readkey()
#     if c == 'x':
#         break
#
#     print(c)


# 퀴즈
# asdw 키를 사용해서 fronze lake로부터
# 목적지까지 캐릭터를 이동하는 코드를 구현하세요
def game_basic():
    env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=False)

    env.reset()
    env.render()

    LEFT, DOWN, RIGHT, UP = 'a', 's', 'd', 'w'

    done = False
    while not done:
        c = readchar.readkey()

        if c != LEFT and c != DOWN and c != RIGHT and c != UP:
            continue

        action = 3
        if c == LEFT:
            action = 0
        elif c == DOWN:
            action = 1
        elif c == RIGHT:
            action = 2

        _, _, done, _, _ = env.step(action)
        env.render()

    env.close()


def game_adv():
    env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=False)

    env.reset()
    env.render()

    LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
    # actions = {'a': LEFT, 's': DOWN, 'd': RIGHT, 'w': UP}
    # actions = {       # 동작 안함
    #     '\x1b[A': UP,
    #     '\x1b[B': DOWN,
    #     '\x1b[C': RIGHT,
    #     '\x1b[D': LEFT
    # }
    actions = {
        '\x00H': UP,
        '\x00P': DOWN,
        '\x00M': RIGHT,
        '\x00K': LEFT
    }

    done = False
    while not done:
        c = readchar.readkey()

        if c not in actions:
            continue

        action = actions[c]

        _, _, done, _, _ = env.step(action)
        env.render()

    env.close()


# game_basic()
game_adv()
