package engine

type Tuple struct {
	a *Value
	b *Value
}

func NewTuple(a *Value, b *Value) *Tuple {
	return &Tuple{a, b}
}

func (t *Tuple) GetA() *Value {
	return t.a
}

func (t *Tuple) GetB() *Value {
	return t.b
}

type Set struct {
	s []Tuple
}

func NewSet() *Set {
	return &Set{s: []Tuple{}}
}

func (s *Set) Add(t *Tuple) {
	// Check if key exists in the Set. If it exists do nothing.
	// If it doesn't exist, add it to the Set.
	for _, v := range s.s {
		if v.GetA() == t.GetA() && v.GetB() == t.GetB() {
			return
		}
		s.s = append(s.s, *t)
	}
}
