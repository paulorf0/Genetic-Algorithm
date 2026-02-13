#pragma once

#include <functional>
#include <iostream>
#include <random>
#include <vector>

template <typename T> using DNA = std::vector<T>;

template <typename T> struct Gene {
  DNA<T> dna;
  double fitness = 0.0;

  bool alive = true;
  int generation = 0;
  int id;
};

template <typename T> using Population = std::vector<Gene<T>>;
template <typename T> using eval = const std::function<double(const DNA<T> &)>;
template <typename T> using gen_rule = const std::function<DNA<T>()>;

/*
 * N = Population Size
 * M = DNA size
 */
template <typename T>
Population<T> initial_pop(const int N, const int M,
                          gen_rule<T> &generation_rule) {
  Population<T> pop;

  for (auto i = 0; i < N; i++) {
    Gene<T> gene;
    for (auto j = 0; j < M; j++) {
      gene.dna = generation_rule();
      gene.generation = 0;
      gene.id = i;
    }

    pop.push_back(gene);
  }

  return pop;
}

template <typename T>
double fitness(const Gene<T> &gene, const eval<T> &evaluator) {

  gene.fitness = evaluator(&gene.dna);

  return gene.fitness;
}

/*
 * N = Number of parents selected.
 */
template <typename T>
std::vector<Gene<T>> selection(int N, const std::vector<Gene<T>> &pop) {
  if (pop.empty())
    return {};

  Population<T> selected_genes;
  selected_genes.reserve(N);

  double total_fitness = 0.0;

  for (const auto &gene : pop) {
    const auto fit = gene.fitness;
    total_fitness += fit;
  }

  if (total_fitness <= 0.0) {
    std::cerr << "Warning: Total fitness is zero or negative. Cannot select."
              << std::endl;
    return {};
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, total_fitness);

  for (auto i = 0; i < N; i++) {
    const auto r = dis(gen);
    double sum = 0;

    for (auto j = 0; j < pop.size(); j++) {
      sum += pop[j].fitness;

      if (r <= sum) {
        selected_genes.push_back(pop[j]);
        break;
      }
    }
  }

  return selected_genes;
}

template <typename T>
Gene<T> crossover(const Gene<T> &parentA, const Gene<T> &parentB) {
  if (parentA.dna.size() != parentA.dna.size()) {
    std::cerr << "ParentA and ParentB size mismatch" << std::endl;
    return {};
  }

  Gene<T> child;
  child.generation = parentA.generation + 1;
  child.dna.reserve(parentA.dna.size());
  // child.id = ??

  std::random_device rd;
  std::mt19937 gen(rd());

  const auto midpoint = gen() % parentA.size();
  for (auto i = 0; i < parentA.dna.size(); i++) {
    if (i <= midpoint) {
      child.dna.push_back(parentA.dna[i]);
    } else {
      child.dna.push_back(parentB.dna[i]);
    }
  }

  return child;
}

template <typename T> void mutate(DNA<T> &dna, double mut) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
  std::normal_distribution<> d(0, 0.1);

  for (auto i = 0; i < dna.size(); i++) {
    if (uniform_dist(gen) < mut) {
      dna[i] += d(gen);
    }
  }
}

template <typename T>
void create_next_generation(Population<T> &pop, const Population<T> &best_genes,
                            double mut) {
  if (mut < 0 || mut > 1) {
    std::cerr << "Warning: mut is lower that 0 or bigger than 1" << std::endl;
    return;
  }

  const int generation = pop[0].generation + 1;
  const int diff = pop.size() - best_genes.size();

  for (auto i = 0; i < best_genes.size(); i++) {
    pop[i] = best_genes[i];
    pop[i].generation = generation;
  }

  std::random_device rd;
  std::mt19937 gen(rd());

  for (auto i = diff; i < pop.size(); i++) {

    const int parentA_idx = gen() % best_genes.size();
    const int parentB_idx = gen() % best_genes.size();

    auto child = crossover(best_genes[parentA_idx], best_genes[parentB_idx]);

    mutate(&child.dna, mut);

    pop[i] = child;
  }
}
