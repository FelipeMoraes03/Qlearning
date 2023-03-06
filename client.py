from subprocess import check_output, check_call
from sys import executable, argv, exit
from random import choice
import polars as pl
import connection as cn

ACTIONS: list = ["left", "right", "jump"]

DIRECTIONS: list = ["north", "east", "south", "west"]


class Qtable:
    file: str
    df: pl.DataFrame

    def __init__(self):
        self.file = "resultado.txt"
        self.df = pl.read_csv(self.file, has_header=False, sep=" ")

    def __get_decoded_state(self, encoded_state: int) -> int:
        platform_mask: int = 0b1111100
        direction_mask: int = 0b0000011

        decoded_state = int(encoded_state, 2)

        platform: int = (decoded_state & platform_mask) >> 2
        direction: int = decoded_state & direction_mask

        return 4 * platform + direction

    def __update(
        self, reward, cur_state, cur_action, learning_rate, discount_factor
    ) -> None:
        new_val = reward + discount_factor * max(self.df[cur_state + 1])
        cur_val = self.df[cur_state][cur_action]

        self.df[cur_state][cur_action] += learning_rate * (new_val - cur_val)

    def execute(
        self, s, num_iterations, cur_platform, learning_rate, discount_factor
    ) -> None:
        cur_state = 4 * cur_platform

        for _ in range(num_iterations):
            optimal_action = max(self.df[cur_state])
            random_action = choice(ACTIONS)

            encoded_state, reward = cn.get_state_reward(s, ACTIONS[optimal_action])
            decoded_state = self.__get_decoded_state(encoded_state)

            self.__update(
                reward,
                cur_state,
                choice([optimal_action, random_action]),
                learning_rate,
                discount_factor,
            )

            cur_state = decoded_state

        self.df.write_csv("resultado.txt", has_header=False, sep=" ")


def main():
    reqs: bytes = check_output([executable, "-m", "pip", "freeze"])
    installed_packages: list = [r.decode().split("==")[0] for r in reqs.split()]
    if "polars" not in installed_packages:
        ans: str = input(
            "The 'polars' API, which is required by this script, \
            is not installed on your system. Do you want to install it? [y/n]"
        )

        if ans == "y":
            check_call(executable, "-m", "pip", "install", "polars")
        elif ans == "n":
            exit()
        else:
            exit("Error: invalid answer")

    if len(argv) != 5:
        exit(
            f"Usage: python3 {argv[0]} [number of iterations] [starting platform] \
             [learning rate] [discount factor]"
        )

    num_iterations = int(argv[1])
    starting_platform = int(argv[2])
    learning_rate = float(argv[3])
    discount_factor = float(argv[4])

    if num_iterations <= 0:
        exit("Error: number of iterations must be positive")

    if learning_rate <= 0 or learning_rate > 1:
        exit(
            "Error: learning rate must be greater than 0 and \
            less than or equal to 1"
        )

    if discount_factor < 0 or discount_factor > 1:
        exit(
            "Error: discount factor must be greater than or equal to 0 and \
            less than or equal to 1"
        )

    s = cn.connect(2037)
    if s != 0:
        table = Qtable()

        while True:
            cmd: str = input(">")

            if cmd == "h":
                print(
                    "Available commands:\n\
                    [h]elp: lists the available commands\n\
                    [e]xecute: executes the Qlearning algorithm\n\
                    [q]uit: terminates the execution of this script"
                )
            elif cmd == "e":
                table.execute(
                    num_iterations, starting_platform, learning_rate, discount_factor
                )
                break
            elif cmd == "q":
                s.close()
                break
            else:
                print("Error: unavailable command (enter 'h' for help)")


if __name__ == "__main__":
    main()
