from subprocess import check_output, check_call
from sys import executable, exit, maxsize
from random import choices
import pandas as pd
import connection as cn

ACTIONS = ["left", "right", "jump"]


class Qtable:
    def __init__(self):
        self.__df = pd.read_csv("resultado.txt", header=None, sep=" ")

    def __get_action(self, cur_state: int) -> int:
        optimal_action: int
        optimal_result: int = -maxsize
        for j in range(3):
            if self.__df[j][cur_state] > optimal_result:
                optimal_action = j
                optimal_result = self.__df[j][cur_state]

        action_probabilities: list = [0.1] * 3
        action_probabilities[optimal_action] = 0.8

        return choices(population=[0, 1, 2], weights=action_probabilities, k=1)[0]

    def __get_decoded_state(self, encoded_state: str) -> int:
        platform = int(encoded_state[2:7], 2)
        direction = int(encoded_state[7:9], 2)

        return 4 * platform + direction

    def __update(
        self,
        reward: int,
        cur_state: int,
        cur_action: int,
        learning_rate: float,
        discount_factor: float,
    ) -> None:
        new_val: int = reward + discount_factor * (self.__df.max(axis=1)[cur_state + 1])
        cur_val: int = self.__df[cur_action][cur_state]

        self.__df[cur_action][cur_state] += learning_rate * (new_val - cur_val)

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

            encoded_state, reward = cn.get_state_reward(s, ACTIONS[cur_action])
            decoded_state: int = self.__get_decoded_state(encoded_state)

            self.__update(
                reward,
                cur_state,
                cur_action,
                learning_rate,
                discount_factor,
            )

            cur_state = decoded_state

        self.__df.to_csv("resultado.txt", header=None, index=None, mode="w", sep=" ")


def main():
    reqs: bytes = check_output([executable, "-m", "pip", "freeze"])
    installed_packages: list = [r.decode().split("==")[0] for r in reqs.split()]
    if "pandas" not in installed_packages:
        ans: str = input(
            "The 'pandas' API, which is required by this script, "
            "is not installed on your system. Do you want to install it [y/n]? "
        )

        if ans == "y":
            check_call([executable, "-m", "pip", "install", "pandas"])
        elif ans == "n":
            exit()
        else:
            exit("Error: invalid answer")

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
                num_iterations = int(input("Number of iterations: "))
                assert (
                    num_iterations > 0
                ), "Error: number of iterations must be positive"

                starting_platform = int(input("Starting platform: "))
                assert (
                    starting_platform >= 1 and starting_platform <= 24
                ), "Error: platforms are numbered from 1 to 24"

                learning_rate = float(input("Learning rate: "))
                assert (
                    learning_rate > 0 and learning_rate <= 1
                ), "Error: learning rate must be greater than 0 "
                "and less than or equal to 1"

                discount_factor = float(input("Discount factor: "))
                assert (
                    discount_factor >= 0 and discount_factor <= 1
                ), "Error: discount factor must be greater than or equal to 0 "
                "and less than or equal to 1"

                table.execute(
                    s, num_iterations, starting_platform, learning_rate, discount_factor
                )
                break
            elif cmd == "q":
                s.close()
                break
            else:
                print("Error: unavailable command (enter 'h' for help)")


if __name__ == "__main__":
    main()
