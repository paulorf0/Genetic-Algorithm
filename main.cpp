#include <iostream>
#include <string>

#include "genetic_algorithm.hpp"
/*
 * Test:
 *   target = 'testando algoritmo'
 *
 */

using dna_type = char;

int main() {
  const std::string alphabet =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ";
  const std::string target = "testando algoritmo";
  const int dna_size = target.size();

  // template <typename T> using Population = std::vector<Gene<T>>;
  // template <typename T> using eval = const std::function<double(const DNA<T>
  // &)>; template <typename T> using gen_rule = const std::function<DNA<T>()>
  // template <typename T> using mut_rule = const std::function<void(T &)>;
  // template <typename T>
  // using ParentPair = std::pair<const Gene<T> &, const Gene<T> &>;
  // template <typename T>
  // using tournament_rule =
  //     const std::function<ParentPair<T>(const Population<T> &)>;

  std::uniform_int_distribution<> dis(0, alphabet.size() - 1);

  mut_rule<char> mutation_rule = [alphabet, &dis](char &param) {
    const auto idx = dis(gen);
    param = alphabet[idx];
  };

  gen_rule<dna_type> generation_rule = [dna_size, alphabet, &dis]() {
    DNA<dna_type> dna;
    dna.reserve(dna_size);

    for (auto i = 0; i < dna_size; ++i) {
      const auto idx = dis(gen);
      dna.push_back(alphabet[idx]);
    }

    return dna;
  };

  eval<dna_type> evaluator = [target](const DNA<dna_type> &dna) {
    double error = 0;

    for (auto i = 0; i < dna.size(); i++) {
      if (dna[i] != target[i])
        error++;
    }

    if (error == 0) {
      return 2.;
    }

    return 1 / error;
  };

  const int N = 50;
  const double accept = 1.5;
  const double mut = 0.015;
  const int epoch = 5000;
  int i = epoch;
  Population<dna_type> new_pop = initial_pop<dna_type>(N, generation_rule);
  Population<dna_type> old_pop = new_pop;
  Population<dna_type> elites;

  const int M = 5;
  while (i-- > 0) {

    for (auto &gene : new_pop) {
      fitness<dna_type>(gene, evaluator);
    }

    elites = selection<dna_type>(M, new_pop);

    old_pop = new_pop;

    if (i % 100 == 0) {
      std::cout << "Epoch: " << epoch - i << std::endl << debug(elites[0]);
    }

    if (elites[0].fitness >= accept) {
      break;
    }

    create_next_generation(new_pop, old_pop, elites, mutation_rule, mut,
                           Rules::Tournament::Tournament_K_best<char>(10));
  }

  auto debug_gene = debug(elites[0]);
  std::cout << "Final: \n" << "Epoch: " << epoch - i << std::endl << debug_gene;
}
