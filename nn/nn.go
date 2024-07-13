package nn

import (
	"math"
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

func (n *Neuron) Call(x []*engine.Value) *engine.Value {
	if len(x) != len(n.Weight) {
		panic("Length of x must be equal to length of n.Weight")
	}
	zipped := make([][]*engine.Value, len(x))
	neuron := NewNeuron(len(x))
	for i := 0; i < len(x); i++ {
		zipped[i] = []*engine.Value{neuron.Weight[i], x[i]}
	}

	activation := engine.NewValue(0.0, "", []*engine.Value{})
	for i := 0; i < len(zipped); i++ {
		activation.Data += zipped[i][0].Data * zipped[i][1].Data
	}
	activation.Data += neuron.Bias.Data
	activation.Data = math.Tanh(activation.Data)
	return activation
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

func (l *Layer) Call(x []*engine.Value) []*engine.Value {
	var outputs []*engine.Value
	for i := 0; i < len(l.Neurons); i++ {
		outputs = append(outputs, l.Neurons[i].Call(x))
	}
	return outputs
}

type MLP struct {
	Layers []*Layer
}

func NewMLP(nin int, nout []int) *MLP {
	_ = len(nout) + 1
	var layers []*Layer
	// input layer
	l := NewLayer(1, nin)
	layers = append(layers, l)
	// hidden and output layers
	for i := 0; i < len(nout); i++ {
		l := NewLayer(nin, nout[i])
		nin = nout[i]
		layers = append(layers, l)
	}
	return &MLP{layers}
}

// func (m *MLP) Call(x []*engine.Value) []*engine.Value {
