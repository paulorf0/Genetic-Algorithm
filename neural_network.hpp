#pragma once

#include <cmath>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <stdexcept>
#include <vector>

using uint = unsigned int;
template <typename T> using vector = std::vector<T>;
/*
 * w^L_{ij} = ligação do neuronio i na camada L+1 com o neuronio j na camada L.
 * b^L_i = viés sobre o neurônio i na camada L+1.
 *
 */

class NeuralNetwork {
private:
  vector<int> size;
  vector<double> weights_bias;

  uint dimension = 0;

  uint getIndexWeight(const uint layer, const uint i, const uint j) {
    if (layer + 1 >= size.size())
      return 0;

    const auto index_layer = getIndexLayer(layer);
    const auto idx = index_layer + i * size[layer] + size[layer + 1];

    return idx;
  }

  uint getIndexBias(const uint layer, const uint i) {
    if (layer + 1 >= size.size())
      return 0;

    const auto index_layer = getIndexLayer(layer);
    const auto idx = index_layer + size[layer] * size[layer + 1] + i;

    return idx;
  }

  inline uint getTotalWeightLayer(const uint layer) {
    if (layer + 1 >= size.size())
      throw std::runtime_error("Error: layer not exist.");

    return size[layer] * size[layer + 1];
  }

  inline uint getTotalBiasLayer(const uint layer) {
    if (layer + 1 >= size.size())
      throw std::runtime_error("Error: layer not exist.");

    return size[layer + 1];
  }

  // layer start at 0..
  // sum starting at 0 until layer - 1
  uint getIndexLayer(const uint layer) {
    if (layer + 1 >= size.size())
      throw std::runtime_error("Error: layer not exist.");

    uint index = 0;
    for (auto i = 0; i < layer; i++)
      index += size[i] * size[i + 1] + size[i + 1];

    return index;
  }

  Eigen::VectorXd activation(Eigen::VectorXd z) {
    return z.unaryExpr([](double z) { return 1.0 / (1.0 + std::exp(-z)); });
  }

public:
  NeuralNetwork(const vector<int> size_) {
    if (size_.empty())
      throw std::runtime_error("Error: Size not initialized");

    size = size_;

    uint total = 0;
    for (auto i = 0; i < size.size(); i++) {
      total += size[i] * size[i + 1] + size[i + 1];
    }

    dimension = total;
    weights_bias.reserve(dimension);
  }

  std::vector<double> forward(vector<double> x) {
    Eigen::VectorXd a = Eigen::Map<Eigen::VectorXd>(x.data(), x.size());

    for (uint l = 0; l < size.size() - 1; l++) {
      const auto idx_l = getIndexLayer(l);
      const auto row = size[l + 1];
      const auto col = size[l];

      if (a.size() != col)
        throw std::runtime_error("Error: Mismatch size");

      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          w(weights_bias.data() + idx_l, row, col);

      Eigen::Map<Eigen::VectorXd> b(weights_bias.data() + idx_l + (row * col),
                                    row);

      auto z = w * a + b;
      a = activation(z);
    }

    return std::vector<double>(a.begin(), a.end());
  }

  // Weights_Bias
  void set_weights_and_bias(vector<double> weights_bias) {
    if (weights_bias.size())
      this->weights_bias = {};

    this->weights_bias = weights_bias;
  }

  uint getDimension() const { return dimension; }
};
