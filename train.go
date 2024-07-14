package main

import (
	"fmt"
	"strings"

	"github.com/ishandhanani/micrograd-go/engine"
	"github.com/ishandhanani/micrograd-go/nn"
)

func main() {
	dummyData()
}

func dummyData() {
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

	// Test performance of neural network on dummy data
	mlp := nn.NewMLP(3, []int{4, 4, 1})

	fmt.Println("\nPredictions vs Observations:")
	fmt.Printf("%-15s %-15s %-15s\n", "Input", "Predicted", "Observed")
	fmt.Println(strings.Repeat("-", 45))

	y_pred := []*engine.Value{}
	for i, x_i := range x {
		pred_i := mlp.Forward(x_i)
		fmt.Printf("(%4.1f, %4.1f, %4.1f) %-15.6f %-15.6f\n",
			x_i[0].Data, x_i[1].Data, x_i[2].Data,
			pred_i[0].Data, y_obs[i].Data)
		y_pred = append(y_pred, pred_i[0])
	}

	// Calculate MSE using engine.Value
	mse := engine.NewValue(0.0, "mse", []*engine.Value{})
	for i := 0; i < len(y_obs); i++ {
		diff := y_obs[i].Subtract(y_pred[i])
		squaredDiff := diff.Multiply(diff)
		mse = mse.Add(squaredDiff)
	}
	mse = mse.Multiply(engine.NewValue(1.0/float64(len(y_obs)), "scale", []*engine.Value{}))

	fmt.Printf("MSE: %f\n", mse.Data)

	// You can now call Backward on mse
	mse.BackwardPass()

	_ = engine.DrawDot(mse, "testmse.png")
}
