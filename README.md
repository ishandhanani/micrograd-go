# Micrograd-Go

This is a port of the [Micrograd](https://github.com/karpathy/micrograd) library to Go. The core of the library is the `Value` type which is a scalar value with a gradient. Using the `Value` type lets us implement backpropagation over a dynamically built directed acyclic graph (DAG) of values and a small neural network library on top. In theory this is enough to train up deep neural networks for regression and classification based tasks (as shown in the `train.go`).

## Usage

### Running the Training Loop

The example in `train.go` shows a full e2e example of a classification task using a multi-layer perceptron (MLP).
To run the training loop, use the following command:

```bash
go run train.go
```

You can enable verbose output with the `-verbose` flag:

```bash
go run train.go -verbose
```

## Structure

```
.
├── engine/
│   ├── engine.go
│   └── engine_test.go
    └── graph.go
├── nn/
│   ├── nn.go
│   └── nn_test.go
├── README.md
├── train.go
├── go.mod
└── go.sum
```

The `engine` package contains the core of the library and implemented the `Value` type which is a scalar value with a gradient. The `nn` package provides a small neural network libary with support for `Neuron`, `Layer`, and `MLP` types. There's also a simple `MSE` loss function implementation. The `graph.go` file contains a helper function to draw the DAG (built using @tmc's [dot](https://github.com/tmc/dot) library).

### Using the Library in Your Project

To use the library in your own Go project, you can import it as follows:

```go
import (
    "github.com/ishandhanani/micrograd-go/engine"
    "github.com/ishandhanani/micrograd-go/nn"
)
```

You can create a simple MLP as follows:

```go
mlp := nn.NewMLP(3, []int{4, 4, 1})
```

This will create an MLP with 3 input neurons, 2 hidden layers of 4 neurons each, and 1 output neuron.

## Visualizing the DAG

To visualize the DAG of a neural network, you can use the `Test_Diagram` function in `engine/engine_test.go`.

To run the tests for the engine and neural network packages, use:

```bash
go test ./...
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgements

This project is based on Andrej Karpathy's micrograd. Check out his [YouTube video](https://www.youtube.com/watch?v=VMj-3S1tku0) for an in-depth explanation of the concepts.
