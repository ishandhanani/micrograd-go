package main

import (
	"flag"
	"fmt"

	"github.com/ishandhanani/micrograd-go/engine"
	"github.com/ishandhanani/micrograd-go/nn"
)

var (
	interactiveMode = flag.Bool("i", false, "Run in interactive mode")
	verboseOutput   = flag.Bool("v", false, "Enable verbose output")
)

func main() {
	flag.Parse()

	if *interactiveMode {
		RunInteractiveMode()
	} else if flag.NFlag() == 0 || (flag.NFlag() == 1 && *verboseOutput) {
		runDefaultTraining(*verboseOutput)
	} else {
		fmt.Println("Invalid flag combination. Use -i for interactive mode or no flags for default training. -v can be used with either mode.")
	}
}

func runDefaultTraining(verbose bool) {
	x := [][]*engine.Value{
		{engine.NewValue(2.0, "x1", nil), engine.NewValue(3.0, "x2", nil), engine.NewValue(-1.0, "x3", nil)},
		{engine.NewValue(3.0, "x1", nil), engine.NewValue(-1.0, "x2", nil), engine.NewValue(0.5, "x3", nil)},
		{engine.NewValue(0.5, "x1", nil), engine.NewValue(1.0, "x2", nil), engine.NewValue(1.0, "x3", nil)},
		{engine.NewValue(1.0, "x1", nil), engine.NewValue(1.0, "x2", nil), engine.NewValue(-1.0, "x3", nil)},
	}
	y_obs := []*engine.Value{
		engine.NewValue(1.0, "y1", nil),
		engine.NewValue(-1.0, "y2", nil),
		engine.NewValue(-1.0, "y3", nil),
		engine.NewValue(1.0, "y4", nil),
	}

	mlp := nn.NewMLP(3, []int{4, 4, 1})
	learningRate := 0.01

	for i := 0; i < 500; i++ {
		y_pred := make([]*engine.Value, len(x))
		for j, x_i := range x {
			y_pred[j] = mlp.Forward(x_i)[0]
		}

		loss := nn.MSE(y_pred, y_obs)
		loss.BackwardPass()

		if verbose || i%100 == 0 || i == 499 {
			fmt.Printf("Iteration %d: Loss: %.6f\n", i, loss.Data)
		}

		if verbose && (i%100 == 0 || i == 499) {
			fmt.Println("Sample predictions:")
			for j := 0; j < len(y_pred) && j < 2; j++ {
				fmt.Printf("  Input %d: Pred: %.6f, Obs: %.6f\n", j, y_pred[j].Data, y_obs[j].Data)
			}
			fmt.Println()
		}

		for _, p := range mlp.Parameters() {
			p.Data = p.Data - (learningRate * p.Grad)
			p.Grad = 0.0
		}
	}

	fmt.Println("\nFinal Predictions:")
	for j, x_i := range x {
		pred := mlp.Forward(x_i)[0].Data
		fmt.Printf("Input %d: Pred: %.6f, Obs: %.6f\n", j, pred, y_obs[j].Data)
	}
}
