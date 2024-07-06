package engine

type Value struct {
	Data float64
}

func (v *Value) Add(v2 *Value) *Value {
	return &Value{v.Data + v2.Data}
}
