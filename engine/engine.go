package engine

import "fmt"

type Value struct {
	Data  float64
	Label string
	prev  []*Value
	op    string
}

func NewValue(data float64, label string, prev []*Value) *Value {
	return &Value{data, label, prev, ""}
}

func (v *Value) Add(v2 *Value) *Value {
	return &Value{Data: v.Data + v2.Data, prev: []*Value{v, v2}, op: "+"}
}

func (v *Value) Multiply(v2 *Value) *Value {
	return &Value{Data: v.Data * v2.Data, prev: []*Value{v, v2}, op: "*"}
}

func (v *Value) String() string {
	return fmt.Sprintf("Value(data=%v)", v.Data)
}

func (v *Value) AddLabel(label string) *Value {
	v.Label = label
	return v
}

func (v *Value) GetPrev() []*Value {
	return v.prev
}

func (v *Value) GetOp() string {
	return v.op
}
