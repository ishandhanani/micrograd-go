package engine

type Value struct {
	Data float64
	Prev *TupleSet
}

func NewValue(data float64, prev TupleSet) *Value {
	return &Value{data, &prev}
}

func (v *Value) Add(v2 *Value) *Value {
	v.Prev.Add(&Tuple{v, v2})
	return &Value{Data: v.Data + v2.Data, Prev: v.Prev}
}

func (v *Value) Multiply(v2 *Value) *Value {
	v.Prev.Add(&Tuple{v, v2})
	return &Value{Data: v.Data * v2.Data, Prev: v.Prev}
}
