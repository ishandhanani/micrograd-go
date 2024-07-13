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
	// note that x must be a slice of length nin
	// [[wi, xi ],...]
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
