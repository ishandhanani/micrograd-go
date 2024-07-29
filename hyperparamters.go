package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/ishandhanani/micrograd-go/engine"
	"github.com/ishandhanani/micrograd-go/nn"
)

func squareAndAdd(x float64) float64 {
	return x*x + 1
}

func generateData(numSamples int) ([][]*engine.Value, []*engine.Value) {
	x := make([][]*engine.Value, numSamples)
	y := make([]*engine.Value, numSamples)

	for i := 0; i < numSamples; i++ {
		input := rand.Float64()*4 - 2 // Random value between -2 and 2
		x[i] = []*engine.Value{engine.NewValue(input, fmt.Sprintf("x%d", i), nil)}
		y[i] = engine.NewValue(squareAndAdd(input), fmt.Sprintf("y%d", i), nil)
	}

	return x, y
}

func getFloatInput(prompt string, defaultValue float64) float64 {
	fmt.Printf("%s (default: %.4f): ", prompt, defaultValue)
	reader := bufio.NewReader(os.Stdin)
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)
	if input == "" {
		return defaultValue
	}
	value, err := strconv.ParseFloat(input, 64)
	if err != nil {
		fmt.Println("Invalid input. Using default value.")
		return defaultValue
	}
	return value
}

func getIntInput(prompt string, defaultValue int) int {
	fmt.Printf("%s (default: %d): ", prompt, defaultValue)
	reader := bufio.NewReader(os.Stdin)
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)
	if input == "" {
		return defaultValue
	}
	value, err := strconv.Atoi(input)
	if err != nil {
		fmt.Println("Invalid input. Using default value.")
		return defaultValue
	}
	return value
}

func getHiddenLayers(prompt string, defaultValue []int) []int {
	fmt.Printf("%s (default: %v): ", prompt, defaultValue)
	reader := bufio.NewReader(os.Stdin)
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)
	if input == "" {
		return defaultValue
	}
	parts := strings.Split(input, ",")
	result := make([]int, len(parts))
	for i, part := range parts {
		value, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil {
			fmt.Println("Invalid input. Using default value.")
			return defaultValue
		}
		result[i] = value
	}
	return result
}

func RunInteractiveMode() {
	fmt.Println("Welcome to the Interactive Neural Network Training!")
	fmt.Println("We'll be training a neural network to approximate the function f(x) = x^2 + 1")
	fmt.Println("Please enter the following hyperparameters (or press Enter for default values):")

	learningRate := getFloatInput("Learning rate", 0.01)
	epochs := getIntInput("Number of epochs", 1000)
	hiddenLayers := getHiddenLayers("Hidden layer sizes (comma-separated)", []int{10, 10})
	numSamples := getIntInput("Number of training samples", 100)
	verbose := getBoolInput("Enable verbose output", false)

	RunHyperparameterTraining(learningRate, epochs, hiddenLayers, numSamples, verbose)
}

func getBoolInput(prompt string, defaultValue bool) bool {
	fmt.Printf("%s (default: %v): ", prompt, defaultValue)
	reader := bufio.NewReader(os.Stdin)
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)
	if input == "" {
		return defaultValue
	}
	value, err := strconv.ParseBool(input)
	if err != nil {
		fmt.Println("Invalid input. Using default value.")
		return defaultValue
	}
	return value
}

func RunHyperparameterTraining(learningRate float64, epochs int, hiddenLayers []int, numSamples int, verbose bool) {
	x, y_obs := generateData(numSamples)

	inputSize := 1
	mlp := nn.NewMLP(inputSize, hiddenLayers)

	fmt.Println("\nTraining the neural network...")

	for i := 0; i < epochs; i++ {
		y_pred := make([]*engine.Value, len(x))
		for j, x_i := range x {
			y_pred[j] = mlp.Forward(x_i)[0]
		}

		loss := nn.MSE(y_pred, y_obs)
		loss.BackwardPass()

		if verbose || i%100 == 0 || i == epochs-1 {
			fmt.Printf("Epoch %d: Loss: %.6f\n", i, loss.Data)
		}

		if verbose && (i%100 == 0 || i == epochs-1) {
			fmt.Println("Sample predictions:")
			for j := 0; j < len(y_pred) && j < 2; j++ {
				fmt.Printf("  Input: %.2f, Pred: %.6f, Actual: %.6f\n", x[j][0].Data, y_pred[j].Data, y_obs[j].Data)
			}
			fmt.Println()
		}

		for _, p := range mlp.Parameters() {
			p.Data = p.Data - (learningRate * p.Grad)
			p.Grad = 0.0
		}
	}

	fmt.Println("\nTraining complete! Let's test the network.")

	testInputs := []float64{-1.5, -0.5, 0, 0.5, 1.5}
	for _, input := range testInputs {
		x_test := []*engine.Value{engine.NewValue(input, "x_test", nil)}
		pred := mlp.Forward(x_test)[0].Data
		actual := squareAndAdd(input)
		fmt.Printf("Input: %.2f, Prediction: %.6f, Actual: %.6f, Absolute Error: %.6f\n",
			input, pred, actual, math.Abs(pred-actual))
	}
}
