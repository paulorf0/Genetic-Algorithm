# NeuroEvolution: Treinamento de Rede Neural via Algoritmo Genético

Este projeto demonstra a implementação de um sistema de **Neuroevolução**, onde os pesos e vieses de uma Rede Neural Artificial (RNA) são otimizados utilizando um Algoritmo Genético (AG), em vez do tradicional *Backpropagation*. O exemplo prático resolve o problema do **XOR** (Ou Exclusivo).

## 🚀 Como Funciona

O projeto combina duas bibliotecas customizadas em C++:

### 1. Rede Neural (`neural_network.hpp`)
* **Arquitetura:** *Feedforward* densa (Multilayer Perceptron).
* **Topologia do Teste:** `[2, 2, 1]` (2 neurônios na entrada, 2 na camada oculta, 1 na saída).
* **Ativação:** Sigmoide.
* **Representação:** A rede aceita um vetor linear (DNA) que é mapeado internamente para as matrizes de pesos e vetores de viés usando a biblioteca `Eigen`.

### 2. Algoritmo Genético (`genetic_algorithm.hpp`)
O AG evolui uma população de vetores de números reais (o DNA da rede) para minimizar o erro quadrático.
* **População:** 200 indivíduos.
* **Seleção:** Torneio (K-Best).
* **Cruzamento (Crossover):** Ponto único (Single Point) com elitismo (Top 5 mantidos).
* **Mutação:** Distribuição Normal adicionada aos genes para exploração estocástica.
* **Fitness:** Inverso do erro quadrático médio em relação às saídas esperadas do XOR.

---

## 📊 Resultados da Execução

Abaixo está o log de uma execução bem-sucedida onde o algoritmo convergiu, encontrando uma solução que satisfaz o problema do XOR.

```text
Final: 
Epoch: 213
Fitness: 10.0889
Id: 116
Dna: ('-4.44953' '3.91753' '-4.55155' '4.83153' '-2.16117' '2.03077' '5.59438' '-4.60265' '1.74622')

Entrada: {0, 0) | Saída da rede: 0.165192 | Esperado: 0
Entrada: {0, 1) | Saída da rede: 0.883073 | Esperado: 1
Entrada: {1, 0) | Saída da rede: 0.805667 | Esperado: 1
Entrada: {1, 1) | Saída da rede: 0.123882 | Esperado: 0
