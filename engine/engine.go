package engine

type Value struct {
	Data float64
	Prev []*Tuple
}

func NewValue(data float64, prev []*Tuple) *Value {
	return &Value{data, prev}
}

func (v *Value) Add(v2 *Value) *Value {
	v.Prev = append(v.Prev, &Tuple{v, v2})
	return &Value{Data: v.Data + v2.Data, Prev: []*Tuple{{v, v2}}}
}

func (v *Value) Multiply(v2 *Value) *Value {
	v.Prev = append(v.Prev, &Tuple{v, v2})
	return &Value{Data: v.Data * v2.Data, Prev: []*Tuple{{v, v2}}}
}
