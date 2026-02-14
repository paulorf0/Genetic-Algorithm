#include <iostream>

#include "genetic_algorithm.hpp"
#include "neural_network.hpp"

int main() {
  const std::vector<int> topology({2, 2, 1});

  NeuralNetwork net(topology);

  std::vector<std::vector<double>> inputs = {
      {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};

  std::vector<double> target = {0.0, 1.0, 1.0, 0.0};

  const int dna_size = net.getDimension();
  using dna_type = double;

  std::uniform_real_distribution<> dis_int(-2.0, 2.0);
  std::normal_distribution<> dis_mut(0.0, 0.1);

  gen_rule<dna_type> generation_rule = [dna_size, &dis_int]() {
    DNA<dna_type> dna;
    dna.reserve(dna_size);

    for (int i = 0; i < dna_size; i++)
      dna.push_back(dis_int(gen));

    return dna;
  };

  mut_rule<dna_type> mutation_rule = [&dis_mut](double &gene) {
    gene += dis_mut(gen);

    if (gene > 10.0)
      gene = 10.0;
    if (gene < -10.0)
      gene = -10.0;
  };

  eval<dna_type> evaluator = [&net, &inputs,
                              &target](const DNA<dna_type> &dna) {
    net.set_weights_and_bias(dna);

    double error = 0;
    for (int i = 0; i < inputs.size(); i++) {
      auto out = net.forward(inputs[i]);

      double diff = out[0] - target[i];
      error += diff * diff;
    }

    return 1.0 / (error + 0.01);
  };

  const int N = 200;
  const double accept = 10.0;
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
                           Rules::Tournament::Tournament_K_best<dna_type>(
                               static_cast<int>(dna_size / 3)));
  }

  auto debug_gene = debug(elites[0]);
  std::cout << "Final: \n"
            << "Epoch: " << epoch - i << std::endl
            << debug_gene;

  for (int i = 0; i < inputs.size(); i++) {
    auto out = net.forward(inputs[i]);
    std::cout << "\n\nEntrada: {" << inputs[i][0] << ", " << inputs[i][1] << ")" << " | Saída da rede: " << out[0]
              << " | Esperado: " << target[i] << std::endl;
  }

  return 0;
}
