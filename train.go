package main

import (
	"fmt"

	"github.com/ishandhanani/micrograd-go/engine"
	"github.com/ishandhanani/micrograd-go/nn"
)

func main() {
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

	for i := 0; i < 20; i++ {
		// forward pass
		y_pred := make([]*engine.Value, len(x))
		for j, x := range x {
			y_pred[j] = mlp.Forward(x)[0] // because we have only one layer
		}
		loss := nn.MSE(y_pred, y_obs)

		// backward pass
		for _, p := range mlp.Parameters() {
			p.Grad = 0.0
		}
		loss.BackwardPass()
		engine.DrawDot(loss, "testloss.png")

		// update parameters
		for _, p := range mlp.Parameters() {
			p.Data += -0.01 * p.Grad
		}

		fmt.Printf("Iteration: %d, Loss: %.4f\n", i, loss.Data)
	}

	// Print the final predictios and the actual values
	y_pred := make([]*engine.Value, len(x))
	for j, x := range x {
		y_pred[j] = mlp.Forward(x)[0] // because we have only one layer
	}
	for i, y := range y_pred {
		fmt.Printf("Predicted: %.4f, Observed: %.4f\n", y.Data, y_obs[i].Data)
	}
}
