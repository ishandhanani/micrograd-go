# Micrograd-Go

Micrograd-go is an simple implementation of the [Micrograd](https://github.com/karpathy/micrograd) library in Go. It contains the same primitives as the original library along with a small neural network library on top. In theory this is enough to train up deep neural networks for regression and classification based tasks (as shown in the `train.go`).

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

### Tests

Both the `engine` and `nn` packages have unit tests. To run the tests for the engine and neural network packages, use:

```bash
go test ./...
```

To visualize the DAG of a neural network, you can use the `Test_Diagram` function in `engine/engine_test.go`.

## Structure

```
.
├── engine/
│   ├── engine.go
│   ├── engine_test.go
    └── graph.go
├── nn/
│   ├── nn.go
│   └── nn_test.go
├── README.md
├── train.go
├── go.mod
└── go.sum
```

The `engine` package contains the core primative of Micrograd-Go: a `Value` which is a scalar value with a gradient. We implement mathematical operations on `Value` object (which gets a bit messy due to Go's lack of operator overloading). The `nn` package provides a small neural network libary with support for the `Neuron`, `Layer`, and `MLP` types. The `graph.go` file contains a helper function to draw the DAG (built using @tmc's [dot](https://github.com/tmc/dot) library).

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgements

This project is based on Andrej Karpathy's micrograd. Check out his [YouTube video](https://www.youtube.com/watch?v=VMj-3S1tku0) for an in-depth explanation of the concepts.
