package nn

import (
	"math/rand"

	"github.com/ishandhanani/micrograd-go/engine"
)

type Neuron struct {
	Weight []*engine.Value // the weights coming into the neuron
	Bias   *engine.Value
}

func NewNeuron(nin int) *Neuron {
	var w []*engine.Value
	var b *engine.Value

	for i := 0; i < nin; i++ {
		w = append(w, engine.NewValue(rand.Float64()*2-1, "w", []*engine.Value{}))
	}

	b = engine.NewValue(rand.Float64()*2-1, "b", []*engine.Value{})

	return &Neuron{w, b}
}

func (n *Neuron) Forward(x []*engine.Value) *engine.Value {
	activation := n.Bias
	for i, input := range x {
		activation = activation.Add(n.Weight[i].Multiply(input))
	}
	return activation.Tanh()
}

func (n *Neuron) Parameters() []*engine.Value {
	return append(n.Weight, n.Bias)
}

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(nin int, nout int) *Layer {
	var neurons []*Neuron
	for i := 0; i < nout; i++ {
		n := NewNeuron(nin)
		neurons = append(neurons, n)
	}
	return &Layer{neurons}
}

func (l *Layer) Forward(x []*engine.Value) []*engine.Value {
	var outputs []*engine.Value
	for i := 0; i < len(l.Neurons); i++ {
		outputs = append(outputs, l.Neurons[i].Forward(x))
	}
	return outputs
}

func (l *Layer) Parameters() []*engine.Value {
	var params []*engine.Value
	for _, n := range l.Neurons {
		params = append(params, n.Parameters()...)
	}
	return params
}

type MLP struct {
	Layers []*Layer
}

func NewMLP(nin int, nout []int) *MLP {
	var layers []*Layer
	layerSizes := append([]int{nin}, nout...)
	for i := 0; i < len(nout); i++ {
		l := NewLayer(layerSizes[i], layerSizes[i+1])
		layers = append(layers, l)
	}
	return &MLP{layers}
}

func (m *MLP) Forward(x []*engine.Value) []*engine.Value {
	for _, layer := range m.Layers {
		new := layer.Forward(x)
		x = new
	}
	return x
}

func (m *MLP) Parameters() []*engine.Value {
	var params []*engine.Value
	for _, layer := range m.Layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}

func MSE(y_pred, y_obs []*engine.Value) *engine.Value {
	if len(y_pred) != len(y_obs) {
		panic("y_pred and y_obs must have the same length")
	}
	sum := engine.NewValue(0.0, "sum", []*engine.Value{})
	for i := 0; i < len(y_obs); i++ {
		diff := y_pred[i].Subtract(y_obs[i])
		squaredDiff := diff.Multiply(diff)
		sum = sum.Add(squaredDiff)
	}
	// Calculate the mean
	n := float64(len(y_obs))
	mean := sum.Multiply(engine.NewValue(1/n, "1/n", []*engine.Value{}))
	return mean.AddLabel("MSE")
}
