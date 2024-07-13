package nn_test

import (
	"fmt"
	"testing"

	"github.com/ishandhanani/micrograd-go/engine"
	"github.com/ishandhanani/micrograd-go/nn"
	"github.com/stretchr/testify/assert"
	//"github.com/stretchr/testify/assert"
)

func Test_Neuron(t *testing.T) {
	n := nn.NewNeuron(2)
	x1 := engine.NewValue(2.0, "x1", []*engine.Value{})
	x2 := engine.NewValue(3.0, "x2", []*engine.Value{})
	x := []*engine.Value{x1, x2}
	y := n.Call(x)
	fmt.Println(y)
}

func Test_Layer(t *testing.T) {
	l := nn.NewLayer(2, 3)
	x1 := engine.NewValue(2.0, "x1", []*engine.Value{})
	x2 := engine.NewValue(3.0, "x2", []*engine.Value{})
	x := []*engine.Value{x1, x2}
	y := l.Call(x)
	fmt.Println(y)
}

func Test_MLP(t *testing.T) {
	x1 := engine.NewValue(2.0, "x1", []*engine.Value{})
	x2 := engine.NewValue(3.0, "x2", []*engine.Value{})
	x3 := engine.NewValue(-1.0, "x3", []*engine.Value{})
	_ = []*engine.Value{x1, x2, x3}
	mlp := nn.NewMLP(3, []int{4, 4, 1})
	// y := mlp.Call(x)
	assert.Equal(t, len(mlp.Layers), 4)
	assert.Equal(t, len(mlp.Layers[0].Neurons), 3)
	assert.Equal(t, len(mlp.Layers[1].Neurons), 4)
	assert.Equal(t, len(mlp.Layers[2].Neurons), 4)
	assert.Equal(t, len(mlp.Layers[2].Neurons), 4)
}
