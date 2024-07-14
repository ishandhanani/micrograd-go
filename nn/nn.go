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
	zipped := make([][]*engine.Value, len(x))
	neuron := NewNeuron(len(x))
	for i := 0; i < len(x); i++ {
		zipped[i] = []*engine.Value{neuron.Weight[i], x[i]}
	}

	activation := engine.NewValue(0.0, "", []*engine.Value{})
	for i := 0; i < len(zipped); i++ {
		activation = activation.Add(zipped[i][0].Multiply(zipped[i][1]))
	}
	activation = activation.Add(neuron.Bias)
	activation = activation.Tanh()
	activation.Data = math.Tanh(activation.Data)
	return activation
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

func (l *Layer) Call(x []*engine.Value) []*engine.Value {
	var outputs []*engine.Value
	for i := 0; i < len(l.Neurons); i++ {
		outputs = append(outputs, l.Neurons[i].Call(x))
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
		new := layer.Call(x)
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
