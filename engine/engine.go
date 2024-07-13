package engine

import (
	"fmt"
	"math"
)

type Value struct {
	Data     float64
	Label    string
	Grad     float64
	prev     []*Value
	op       string
	Backward func()
}

func NewValue(data float64, label string, prev []*Value) *Value {
	return &Value{data, label, 0.0, prev, "", nil}
}

func (v *Value) Add(v2 *Value) *Value {
	out := &Value{Data: v.Data + v2.Data, prev: []*Value{v, v2}, op: "+"}
	out.Backward = func() {
		v.Grad += 1.0 * out.Grad
		v2.Grad += 1.0 * out.Grad
	}
	out.Backward()
	return out
}

func Negate(v *Value) *Value {
	v.Data = v.Data * -1
	return v
}

func (v *Value) Subtract(v2 *Value) *Value {
	negated := Negate(v2)
	return v.Add(negated)
}

func (v *Value) Multiply(v2 *Value) *Value {
	out := &Value{Data: v.Data * v2.Data, prev: []*Value{v, v2}, op: "*"}
	out.Backward = func() {
		v.Grad += out.Grad * v2.Data
		v2.Grad += out.Grad * v.Data
	}
	out.Backward()
	return out
}

func Pow(v *Value, v2 *Value) *Value {
	out := &Value{Data: math.Pow(v.Data, v2.Data), prev: []*Value{v, v2}, op: "^"}
	out.Backward = func() {
		v.Grad += (v2.Data * math.Pow(v.Data, v2.Data-1)) * out.Grad
		v2.Grad += (v.Data * math.Pow(v.Data, v2.Data)) * out.Grad
	}
	out.Backward()
	return out
}

func (v *Value) SimpleTanh() *Value {
	out := &Value{Data: math.Tanh(v.Data), prev: []*Value{v}, op: "tanh"}
	out.Backward = func() {
		v.Grad += (1 - (v.Data * v.Data)) * out.Grad
	}
	out.Backward()
	return out
}

func (v *Value) Tanh() *Value {
	out := &Value{Data: (math.Exp(2*v.Data) - 1) / (math.Exp(2*v.Data) + 1), prev: []*Value{v}, op: "tanh"}
	out.Backward = func() {
		v.Grad += (1 - math.Pow(out.Data, 2)) * out.Grad
	}
	out.Backward()
	return out
}

func Exp(v *Value) *Value {
	out := &Value{Data: math.Exp(v.Data), prev: []*Value{v}, op: "exp"}
	out.Backward = func() {
		v.Grad += out.Grad * out.Data
	}
	out.Backward()
	return out
}

func (v *Value) String() string {
	return fmt.Sprintf("Value(data=%v)", v.Data)
}

func (v *Value) GetPrev() []*Value {
	return v.prev
}

func (v *Value) GetOp() string {
	return v.op
}
func (v *Value) AddLabel(label string) *Value {
	v.Label = label
	return v
}

func TopologicalSort(root *Value) []*Value {
	sorted := []*Value{}
	visited := make(map[*Value]bool)

	var topo func(*Value)
	topo = func(v *Value) {
		if !visited[v] {
			visited[v] = true
			for _, child := range v.prev {
				topo(child)
			}
			sorted = append(sorted, v)
		}
	}
	topo(root)

	return sorted
}

func (v *Value) BackwardPass() {
	topo := []*Value{}
	visited := map[*Value]bool{}

	topo = buildTopo(v, topo, visited)

	v.Grad = 1.0

	for i := len(topo) - 1; i >= 0; i-- {
		if len(topo[i].prev) != 0 { // added this because there are multiple variations of topological sort for a single graph
			topo[i].Backward()
		}
	}
}

func buildTopo(v *Value, topo []*Value, visited map[*Value]bool) []*Value {
	if !visited[v] {
		visited[v] = true
		for _, prev := range v.prev {
			topo = buildTopo(prev, topo, visited)
		}
		topo = append(topo, v)
	}
	return topo
}
