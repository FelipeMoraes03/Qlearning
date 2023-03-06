from subprocess import check_output, check_call
from sys import executable, argv, exit, maxsize
from random import choices
import polars as pl
import connection as cn

class Qtable:
    def __init__(self):
        self.__df: pl.DataFrame = pl.read_csv(
            "resultado.txt", has_header=False, sep=" "
        )

    def __get_action(self, cur_state: int) -> int:
        optimal_action: int
        optimal_result: int = -maxsize
        for j in range(3):
            if self.__df[cur_state][j] > optimal_result:
                optimal_action = j
                optimal_result = self.__df[cur_state][j]

        action_probabilities: list = [0.1] * 3
        action_probabilities[optimal_action] = 0.8

        return choices(
            population=[0, 1, 2], weights=action_probabilities, k=1
        )[0]

    def __get_decoded_state(self, encoded_state: str) -> int:
        platform_mask = 0b1111100
        direction_mask = 0b0000011

        decoded_state = int(encoded_state, 2)

        platform: int = (decoded_state & platform_mask) >> 2
        direction: int = decoded_state & direction_mask

        return 4 * platform + direction

    def __update(
        self,
        reward: int,
        cur_state: int,
        cur_action: int,
        learning_rate: float,
        discount_factor: float,
    ) -> None:
        new_val: int = reward + discount_factor * max(self.__df[cur_state + 1])
        cur_val: int = self.__df[cur_state][cur_action]

        self.__df[cur_state][cur_action] += learning_rate * (new_val - cur_val)

    def execute(
        self,
        s: cn.socket.socket,
        num_iterations: int,
        cur_platform: int,
        learning_rate: float,
        discount_factor: float,
    ) -> None:
        cur_state: int = 4 * cur_platform

        for _ in range(num_iterations):
            cur_action: int = self.__get_action(cur_state)

            encoded_state, reward = cn.get_state_reward(s, cur_action)
            decoded_state: int = self.__get_decoded_state(encoded_state)

            self.__update(
                reward,
                cur_state,
                cur_action,
                learning_rate,
                discount_factor,
            )

            cur_state = decoded_state

        self.__df.write_csv("resultado.txt", has_header=False, sep=" ")


def main():
    reqs: bytes = check_output([executable, "-m", "pip", "freeze"])
    installed_packages: list = [r.decode().split("==")[0] for r in reqs.split()]
    if "polars" not in installed_packages:
        ans: str = input(
            "The 'polars' API, which is required by this script, "
            "is not installed on your system. Do you want to install it [y/n]? "
        )

        if ans == "y":
            check_call([executable, "-m", "pip", "install", "polars"])
        elif ans == "n":
            exit()
        else:
            exit("Error: invalid answer")

    if len(argv) != 5:
        exit(
            f"Usage: python3 {argv[0]} [number of iterations] "
            "[starting platform] [learning rate] [discount factor]"
        )

    num_iterations = int(argv[1])
    starting_platform = int(argv[2])
    learning_rate = float(argv[3])
    discount_factor = float(argv[4])

    if num_iterations <= 0:
        exit("Error: number of iterations must be positive")

    if learning_rate <= 0 or learning_rate > 1:
        exit(
            "Error: learning rate must be greater than 0 "
            "and less than or equal to 1"
        )

    if discount_factor < 0 or discount_factor > 1:
        exit(
            "Error: discount factor must be greater than or equal to 0 and "
            "less than or equal to 1"
        )

    s: cn.socket.socket or int = cn.connect(2037)
    if s != 0:
        table = Qtable()

        while True:
            cmd: str = input(">")

            if cmd == "h":
                print(
                    "Available commands:\n"
                    "[h]elp: lists the available commands\n"
                    "[e]xecute: executes the Qlearning algorithm\n"
                    "[q]uit: terminates the execution of this script"
                )
            elif cmd == "e":
                table.execute(
                    num_iterations, starting_platform, learning_rate,
                    discount_factor
                )
                break
            elif cmd == "q":
                s.close()
                break
            else:
                print("Error: unavailable command (enter 'h' for help)")


if __name__ == "__main__":
    main()
