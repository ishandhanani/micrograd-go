package main

import (
	"flag"
	"fmt"

	"github.com/ishandhanani/micrograd-go/engine"
	"github.com/ishandhanani/micrograd-go/nn"
)

func main() {
	verbose := flag.Bool("verbose", false, "Enable verbose output")
	flag.Parse()

	// Generate random data
	x := [][]*engine.Value{
		{engine.NewValue(2.0, "x1", []*engine.Value{}), engine.NewValue(3.0, "x2", []*engine.Value{}), engine.NewValue(-1.0, "x3", []*engine.Value{})},
		{engine.NewValue(3.0, "x1", []*engine.Value{}), engine.NewValue(-1.0, "x2", []*engine.Value{}), engine.NewValue(0.5, "x3", []*engine.Value{})},
		{engine.NewValue(0.5, "x1", []*engine.Value{}), engine.NewValue(1.0, "x2", []*engine.Value{}), engine.NewValue(1.0, "x3", []*engine.Value{})},
		{engine.NewValue(1.0, "x1", []*engine.Value{}), engine.NewValue(1.0, "x2", []*engine.Value{}), engine.NewValue(-1.0, "x3", []*engine.Value{})},
	}
	y_obs := []*engine.Value{
		engine.NewValue(1.0, "y1", []*engine.Value{}),
		engine.NewValue(-1.0, "y2", []*engine.Value{}),
		engine.NewValue(-1.0, "y3", []*engine.Value{}),
		engine.NewValue(1.0, "y4", []*engine.Value{}),
	}

	mlp := nn.NewMLP(3, []int{4, 4, 1})
	learningRate := 0.01

	for i := 0; i < 20; i++ {
		// Forward pass
		y_pred := make([]*engine.Value, len(x))
		for j, x_i := range x {
			y_pred[j] = mlp.Forward(x_i)[0]
		}

		// Compute loss
		loss := nn.MSE(y_pred, y_obs)

		// Backward pass
		loss.BackwardPass()

		// Always print iteration number and loss
		fmt.Printf("Iteration %d: Loss: %.6f\n", i, loss.Data)

		if *verbose {
			fmt.Println("Predictions vs Observations:")
			for j := range y_pred {
				fmt.Printf("  Pred: %.6f, Obs: %.6f\n", y_pred[j].Data, y_obs[j].Data)
			}

			fmt.Println("Sample gradients and updates:")
			for j, p := range mlp.Parameters() {
				if j < 5 { // Print first 5 parameters
					update := -learningRate * p.Grad
					fmt.Printf("  Param: %.6f, Grad: %.6f, Update: %.6f\n", p.Data, p.Grad, update)
				}
			}
			fmt.Println() // Add a blank line for readability in verbose mode
		}

		// Update parameters
		for _, p := range mlp.Parameters() {
			p.Data += -learningRate * p.Grad
		}
	}

	// Always print final predictions
	fmt.Println("\nFinal Predictions:")
	for j, x_i := range x {
		pred := mlp.Forward(x_i)[0].Data
		fmt.Printf("Input %d: Pred: %.6f, Obs: %.6f\n", j, pred, y_obs[j].Data)
	}
}
