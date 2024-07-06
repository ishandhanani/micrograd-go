package engine

type Tuple struct {
	curr *Value
	prev *Value
}

func NewTuple(a *Value, b *Value) *Tuple {
	return &Tuple{a, b}
}

func (t *Tuple) GetCurr() *Value {
	return t.curr
}

func (t *Tuple) GetPrev() *Value {
	return t.prev
}

type TupleSet struct {
	s []Tuple
}

func NewSet() *TupleSet {
	return &TupleSet{s: []Tuple{}}
}

func (s *TupleSet) Add(t *Tuple) {
	// Check if key exists in the Set. If it exists do nothing.
	// If it doesn't exist, add it to the Set.
	for _, v := range s.s {
		if v.GetCurr() == t.GetCurr() && v.GetPrev() == t.GetPrev() {
			return
		}
		s.s = append(s.s, *t)
	}
}
