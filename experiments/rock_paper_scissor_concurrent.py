import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import random


def simulate_games(num_games, a_s, a_r, b_p, b_r, b_s):
    scores = {"A": 0, "B": 0}

    for _ in range(num_games):
        a_move = random.choices(["Scissors", "Rock"], [a_s, a_r], k=1)[0]
        b_move = random.choices(["Scissors", "Rock", "Paper"], [b_s, b_r, b_p], k=1)[0]

        if a_move == b_move:
            continue
        elif (a_move == "Rock" and b_move == "Scissors") or (
            a_move == "Scissors" and b_move == "Paper"
        ):
            scores["A"] += 1
            scores["B"] -= 1
        else:
            scores["A"] -= 1
            scores["B"] += 1

    return scores


def find_best_a(cpus, num_games=1000):
    best_a_s = best_a_r = 0
    with Pool(cpus) as p:
        min_max_score = float("inf")
        for a_s in tqdm(np.arange(0, 1.01, 0.01), desc="Calculating best strategies for A"):
            a_r = 1 - a_s
            scores_list = []
            for b_p in np.arange(0, 1.01, 0.01):
                for b_r in np.arange(0, 1.01 - b_p, 0.01):
                    b_s = 1 - b_p - b_r
                    args_list = [(num_games, a_s, a_r, b_p, b_r, b_s) for _ in range(cpus)]
                    scores_list.extend(p.starmap(simulate_games, args_list))
            max_b_score = max(scores["B"] for scores in scores_list)
            if max_b_score < min_max_score:
                min_max_score = max_b_score
                best_a_s = a_s
                best_a_r = a_r
    return best_a_r, best_a_s


def find_best_b(cpus, num_games=1000):
    with Pool(cpus) as p:
        best_b_s = best_b_r = best_b_p = 0
        min_max_score = float("inf")
        for b_s in tqdm(np.arange(0, 1.01, 0.01), desc="Calculating best strategies for B"):
            for b_r in np.arange(0, 1.01, 0.01):
                b_p = 1 - b_s - b_r
                scores_list = []
                for a_s in np.arange(0, 1.01, 0.01):
                    a_r = 1 - a_s
                    args_list = [
                        (num_games, a_s, a_r, b_p, b_r, b_s) for _ in range(cpus)
                    ]
                    scores_list.extend(p.starmap(simulate_games, args_list))
                max_a_score = max(scores["A"] for scores in scores_list)
                if max_a_score < min_max_score:
                    min_max_score = max_a_score
                    best_b_s = b_s
                    best_b_r = b_r
                    best_b_p = b_p
    return best_b_p, best_b_r, best_b_s


def main():
    cpus = multiprocessing.cpu_count()
    print(f"Using {cpus} CPUs")

    best_a_r, best_a_s = find_best_a(cpus,num_games=1000)
    print(f"Best strategy for A: Scissors = {best_a_s}, Rock = {best_a_r}")

    # best_b_p, best_b_r, best_b_s = find_best_b(cpus, 1000)
    # print(
    #     f"Best strategy for B: Scissors = {best_b_s}, Rock = {best_b_r}, Paper = {best_b_p}"
    # )


if __name__ == "__main__":
    main()
