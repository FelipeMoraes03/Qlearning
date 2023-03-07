from subprocess import check_output, check_call
from sys import executable, exit, maxsize
from random import choices
import pandas as pd
import connection as cn

ACTIONS = ["left", "right", "jump"]


class Qtable:
    def __init__(self):
        self.__df = pd.read_csv("tableBase.txt", header=None, sep=" ")
        self.__victory = 0
        self.__death = 0

    def get_victories(self):
        return self.__victory
    
    def get_deaths(self):
        return self.__death

    def __get_action(self, cur_state: int) -> int:
        """
        There are 3 possible actions in each state: left, right and jump.
        One of them is optimal and has a probability of 80%.
        The rest of them have a probability of 10% each.

        This method draws an action and returns it.
        """

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
        """
        Each state is encoded as a bit vector, where the 5 MSBs designate the current platform,
        and the 2 LSBs designate the current action.

        This method decodes the state into a decimal integer and returns it.
        """

        platform = int(encoded_state[2:7], 2)
        direction = int(encoded_state[7:9], 2)

        return 4 * platform + direction

    def __update(
        self,
        reward: int,
        cur_state: int,
        next_state: int,
        cur_action: int,
        learning_rate: float,
        discount_factor: float,
    ) -> None:
        """
        This method uses the Bellman equation to update this table's DataFrame,
        given the current state and the current action.
        """

        new_val: int = reward + discount_factor * (self.__df.max(axis=1)[next_state])
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
        """
        This method executes the Qlearning algorithm on this table.
        """

        # Rationale: multiplying by 4 is the same as shifting 2 bits to the left
        cur_state: int = 4 * cur_platform

        for _ in range(num_iterations):
            cur_action: int = self.__get_action(cur_state)

            encoded_state, reward = cn.get_state_reward(s, ACTIONS[cur_action])
            decoded_state: int = self.__get_decoded_state(encoded_state)

            if reward == 300: 
                self.__victory += 1
            elif reward == -100: 
                self.__death +=1

            self.__update(
                reward,
                cur_state,
                decoded_state,
                cur_action,
                learning_rate,
                discount_factor,
            )

            cur_state = decoded_state

        self.__df.to_csv("resultado.txt", header=None, index=None, mode="w", sep=" ")


def main():
    # Checks if the 'pandas' API is installed. If not, suggests installing it
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
                    starting_platform >= 0 and starting_platform <= 23
                ), "Error: platforms are numbered from 0 to 23"

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

    print(f"\nVictories: {table.get_victories()}")
    print(f"Deaths: {table.get_deaths()}")

if __name__ == "__main__":
    main()