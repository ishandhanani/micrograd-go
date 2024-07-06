package engine

type children struct {
	current *Value // in the video this is self
	other   *Value // in the video this is other
}
type Value struct {
	Data float64
	prev []*children // in the video this is a set of children tuples
}

func NewValue(data float64, prev []*children) *Value {
	return &Value{data, prev}
}

func (v *Value) Add(v2 *Value) *Value {
	return &Value{Data: v.Data + v2.Data, prev: []*children{{current: v, other: v2}}}
}

func (v *Value) Multiply(v2 *Value) *Value {
	return &Value{Data: v.Data * v2.Data, prev: []*children{{current: v, other: v2}}}
}
