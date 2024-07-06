package engine

type Tuple struct {
	curr *Value
	next *Value
}

func NewTuple(a *Value, b *Value) *Tuple {
	return &Tuple{a, b}
}

func (t *Tuple) GetCurr() *Value {
	return t.curr
}

func (t *Tuple) GetNext() *Value {
	return t.next
}
