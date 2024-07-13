package nn_test

import (
	"fmt"
	"testing"

	"github.com/ishandhanani/micrograd-go/engine"
	"github.com/ishandhanani/micrograd-go/nn"
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
