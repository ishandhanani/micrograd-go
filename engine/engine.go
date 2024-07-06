package engine

type Value struct {
	Data float64
	Prev []*Tuple
	Op   string
}

func NewValue(data float64, prev []*Tuple, op string) *Value {
	return &Value{data, prev, op}
}

func (v *Value) Add(v2 *Value) *Value {
	v.Prev = append(v.Prev, &Tuple{v, v2})
	return &Value{Data: v.Data + v2.Data, Prev: []*Tuple{{v, v2}}, Op: "+"}
}

func (v *Value) Multiply(v2 *Value) *Value {
	v.Prev = append(v.Prev, &Tuple{v, v2})
	return &Value{Data: v.Data * v2.Data, Prev: []*Tuple{{v, v2}}, Op: "*"}
}
