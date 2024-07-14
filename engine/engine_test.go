package engine_test

import (
	"fmt"
	"testing"

	"github.com/ishandhanani/micrograd-go/engine"
	"github.com/stretchr/testify/assert"
)

func Test_Add(t *testing.T) {
	v1 := engine.NewValue(-2.0, "", []*engine.Value{})
	v2 := engine.NewValue(3.0, "", []*engine.Value{})
	v3 := v1.Add(v2)
	if v3.Data != 1 {
		t.Errorf("Expected %v, got %v", 1, v3.Data)
	}
}

func Test_Multiply(t *testing.T) {
	v1 := engine.NewValue(-2.0, "", []*engine.Value{})
	v2 := engine.NewValue(3.0, "", []*engine.Value{})
	v3 := v1.Multiply(v2)
	if v3.Data != -6 {
		t.Errorf("Expected %v, got %v", -6, v3.Data)
	}
}

func Test_VideoExample1(t *testing.T) {
	a := engine.Value{Data: 2.0}
	b := engine.Value{Data: -3.0}
	c := engine.Value{Data: 10.0}
	d := a.Multiply(&b).Add(&c)

	assert.Equal(t, d.Data, 4.0)
}

func Test_Diagram(t *testing.T) {
	a := engine.NewValue(-2.0, "a", []*engine.Value{})
	b := engine.NewValue(3.0, "b", []*engine.Value{})
	c := engine.NewValue(10.0, "c", []*engine.Value{})
	e := a.Multiply(b).AddLabel("e")
	d := e.Add(c).AddLabel("d")
	f := engine.NewValue(-2.0, "f", []*engine.Value{})
	L := d.Multiply(f).AddLabel("L")

	engine.DrawDot(L, "testdiagram.png")
}

/**
 * For completeness - heres how the Pytorch code looks like for the same setup
 * x1 = torch.Tensorr([2.0]).requires_grad_(True)
 * x2 = torch.Tensorr([0.0]).requires_grad_(True)
 * w1 = torch.Tensorr([-3.0]).requires_grad_(True)
 * w2 = torch.Tensorr([1.0]).requires_grad_(True)
 * b = torch.Tensorr([6.7]).requires_grad_(True)
 * n = x1*x2 + w1*w2 + b
 * o = torch.tanh(n)
 * o.backward()
 */

func SetupNeuralNetwork() (x1, x2, w1, w2, b *engine.Value) {
	x1 = engine.NewValue(2.0, "x1", []*engine.Value{})
	x2 = engine.NewValue(0.0, "x2", []*engine.Value{})
	// weights w1, w2
	w1 = engine.NewValue(-3.0, "w1", []*engine.Value{})
	w2 = engine.NewValue(1.0, "w2", []*engine.Value{})
	// neuron bias
	b = engine.NewValue(6.7, "b", []*engine.Value{})

	return x1, x2, w1, w2, b
}

func Test_NeuronBackpropExample(t *testing.T) {
	x1, x2, w1, w2, b := SetupNeuralNetwork()

	x1w1 := x1.Multiply(w1).AddLabel("x1w1")
	x2w2 := x2.Multiply(w2).AddLabel("x2w2")
	x1w1x2w2 := x1w1.Add(x2w2).AddLabel("x1w1 + x2w2")
	n := x1w1x2w2.Add(b).AddLabel("n")

	o := n.Tanh().AddLabel("o")

	// manual global derivative
	o.Grad = 1.0

	// manual backward pass
	o.Backward()
	n.Backward()
	x1w1x2w2.Backward()
	x1w1.Backward()
	x2w2.Backward()

	engine.DrawDot(o, "testneuralbackprop.png")
}

func Test_TopologicalSort(t *testing.T) {
	x1, x2, w1, w2, b := SetupNeuralNetwork()

	x1w1 := x1.Multiply(w1).AddLabel("x1w1")
	x2w2 := x2.Multiply(w2).AddLabel("x2w2")
	x1w1x2w2 := x1w1.Add(x2w2).AddLabel("x1w1 + x2w2")
	n := x1w1x2w2.Add(b).AddLabel("n")

	o := n.Tanh().AddLabel("o")
	o.Grad = 1.0

	sorted := engine.TopologicalSort(o)

	fmt.Println(sorted)
}

func Test_AutomaticBackpropagation(t *testing.T) {
	x1, x2, w1, w2, b := SetupNeuralNetwork()

	x1w1 := x1.Multiply(w1).AddLabel("x1w1")
	x2w2 := x2.Multiply(w2).AddLabel("x2w2")
	x1w1x2w2 := x1w1.Add(x2w2).AddLabel("x1w1 + x2w2")
	n := x1w1x2w2.Add(b).AddLabel("n")

	o := n.Tanh().AddLabel("o")
	o.Grad = 1.0

	// automatic backward pass
	o.BackwardPass()

	engine.DrawDot(o, "testautomaticbackprop.png")
}
