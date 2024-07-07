package engine

import "fmt"

type Value struct {
	Data float64
	Prev []*Value
	op   string
}

func NewValue(data float64, prev []*Value, op string) *Value {
	return &Value{data, prev, op}
}

func (v *Value) Add(v2 *Value) *Value {
	return &Value{Data: v.Data + v2.Data, Prev: []*Value{v, v2}, op: "+"}
}

func (v *Value) Multiply(v2 *Value) *Value {
	return &Value{Data: v.Data * v2.Data, Prev: []*Value{v, v2}, op: "*"}
}

func (v *Value) String() string {
	return fmt.Sprintf("Value(data=%v)", v.Data)
}
