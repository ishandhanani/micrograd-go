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
	backward func(v *Value)
}

func NewValue(data float64, label string, prev []*Value) *Value {
	return &Value{data, label, 0.0, prev, "", nil}
}

func (v *Value) Add(v2 *Value) *Value {
	return &Value{Data: v.Data + v2.Data, prev: []*Value{v, v2}, op: "+"}
}

func (v *Value) Multiply(v2 *Value) *Value {
	return &Value{Data: v.Data * v2.Data, prev: []*Value{v, v2}, op: "*"}
}

func (v *Value) Tanh() *Value {
	return &Value{Data: math.Tanh(v.Data), prev: []*Value{v}, op: "tanh"}
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
