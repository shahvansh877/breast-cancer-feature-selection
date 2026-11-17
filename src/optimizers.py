
import numpy as np


def random_mask(n, rng):
    mask = rng.random(n) < 0.5
    if not mask.any():
        mask[rng.integers(n)] = True
    return mask


def repair(mask, rng):
    mask = np.asarray(mask, dtype=bool).copy()
    if not mask.any():
        mask[rng.integers(len(mask))] = True
    return mask


def simulated_annealing(fitness, n_features, iterations=10, seed=42):
    rng = np.random.default_rng(seed)
    current = random_mask(n_features, rng)
    current_score = fitness.evaluate(current)

    best = current.copy()
    best_score = current_score
    history = []

    temperature = 0.1

    for step in range(iterations):
        candidate = current.copy()
        candidate[rng.integers(n_features)] ^= True
        candidate = repair(candidate, rng)

        candidate_score = fitness.evaluate(candidate)
        delta = candidate_score - current_score

        if delta > 0 or rng.random() < np.exp(
            max(-700, delta / temperature)
        ):
            current = candidate
            current_score = candidate_score

        if current_score > best_score:
            best = current.copy()
            best_score = current_score

        history.append(best_score)
        temperature *= 0.85

    return best, history


def genetic_algorithm(
    fitness, n_features, population_size=8,
    generations=10, mutation_rate=0.05, seed=42
):
    rng = np.random.default_rng(seed)

    population = np.array([
        random_mask(n_features, rng)
        for _ in range(population_size)
    ])

    history = []
    best = None
    best_score = -np.inf

    def score(mask):
        return fitness.evaluate(mask)

    for _ in range(generations):
        scores = np.array([score(ind) for ind in population])

        idx = np.argmax(scores)
        if scores[idx] > best_score:
            best = population[idx].copy()
            best_score = scores[idx]

        # Preserve the two best individuals.
        elite_idx = np.argsort(scores)[-2:]
        new_population = [population[i].copy() for i in elite_idx]

        def tournament():
            ids = rng.choice(population_size, size=3, replace=True)
            winner = ids[np.argmax(scores[ids])]
            return population[winner].copy()

        while len(new_population) < population_size:
            p1, p2 = tournament(), tournament()
            point = rng.integers(1, n_features)

            child1 = np.concatenate([p1[:point], p2[point:]])
            child2 = np.concatenate([p2[:point], p1[point:]])

            for child in (child1, child2):
                flips = rng.random(n_features) < mutation_rate
                child[flips] = ~child[flips]
                new_population.append(repair(child, rng))

                if len(new_population) >= population_size:
                    break

        population = np.array(new_population[:population_size])
        history.append(best_score)

    return best, history


def binary_pso(
    fitness, n_features, population_size=8,
    iterations=10, w=0.7, c1=1.5, c2=1.5, seed=42
):
    rng = np.random.default_rng(seed)

    positions = np.array([
        random_mask(n_features, rng)
        for _ in range(population_size)
    ])

    velocities = rng.uniform(
        -1, 1, size=(population_size, n_features)
    )

    pbest = positions.copy()
    pbest_scores = np.array([
        fitness.evaluate(p) for p in positions
    ])

    idx = np.argmax(pbest_scores)
    gbest = pbest[idx].copy()
    gbest_score = pbest_scores[idx]
    history = []

    for _ in range(iterations):
        for i in range(population_size):
            r1 = rng.random(n_features)
            r2 = rng.random(n_features)

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest[i].astype(float)
                             - positions[i].astype(float))
                + c2 * r2 * (gbest.astype(float)
                             - positions[i].astype(float))
            )

            velocities[i] = np.clip(velocities[i], -6, 6)
            probability = 1 / (1 + np.exp(-velocities[i]))

            positions[i] = rng.random(n_features) < probability
            positions[i] = repair(positions[i], rng)

            score = fitness.evaluate(positions[i])

            if score > pbest_scores[i]:
                pbest[i] = positions[i].copy()
                pbest_scores[i] = score

            if score > gbest_score:
                gbest = positions[i].copy()
                gbest_score = score

        history.append(gbest_score)

    return gbest, history


def firefly_algorithm(
    fitness, n_features, population_size=8,
    iterations=10, beta0=1.0, gamma=1.0,
    mutation_rate=0.05, seed=42
):
    rng = np.random.default_rng(seed)

    population = np.array([
        random_mask(n_features, rng)
        for _ in range(population_size)
    ])

    scores = np.array([
        fitness.evaluate(p) for p in population
    ])

    idx = np.argmax(scores)
    best = population[idx].copy()
    best_score = scores[idx]
    history = []

    for _ in range(iterations):
        for i in range(population_size):
            for j in range(population_size):
                if scores[j] <= scores[i]:
                    continue

                # Normalized Hamming distance.
                distance = np.mean(population[i] != population[j])
                attraction = beta0 * np.exp(
                    -gamma * distance ** 2
                )

                candidate = population[i].copy()
                differs = candidate != population[j]

                move = (
                    rng.random(n_features) < attraction
                ) & differs

                candidate[move] = population[j][move]

                # Random exploration.
                flips = rng.random(n_features) < mutation_rate
                candidate[flips] = ~candidate[flips]
                candidate = repair(candidate, rng)

                candidate_score = fitness.evaluate(candidate)

                if candidate_score > scores[i]:
                    population[i] = candidate
                    scores[i] = candidate_score

                    if candidate_score > best_score:
                        best = candidate.copy()
                        best_score = candidate_score

        history.append(best_score)

    return best, history
