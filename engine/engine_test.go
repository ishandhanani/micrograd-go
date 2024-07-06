package engine_test

import (
	"testing"

	"github.com/ishandhanani/micrograd-go/engine"
	"github.com/stretchr/testify/assert"
)

func TestAdd(t *testing.T) {
	v1 := &engine.Value{Data: 1, Prev: []*engine.Tuple{}}
	v2 := &engine.Value{Data: 2, Prev: []*engine.Tuple{}}
	v3 := v1.Add(v2)
	if v3.Data != 3 {
		t.Errorf("Expected %v, got %v", 3, v3.Data)
	}
}

func TestMultiply(t *testing.T) {
	v1 := &engine.Value{Data: 1, Prev: []*engine.Tuple{}}
	v2 := &engine.Value{Data: 2, Prev: []*engine.Tuple{}}
	v3 := v1.Add(v2)
	if v3.Data != 3 {
		t.Errorf("Expected %v, got %v", 3, v3.Data)
	}
}

func TestVideoExample1(t *testing.T) {
	// a = 2
	// b = -3
	// c = 10
	// d._prev = {Value(-6), Value(10)}
	a := engine.Value{Data: 2.0}
	b := engine.Value{Data: -3.0}
	c := engine.Value{Data: 10.0}
	d := a.Multiply(&b).Add(&c)

	assert.Equal(t, d.Data, 4.0)

	assert.Equal(t, d.Prev[0].GetCurr().Data, -6.0)
	assert.Equal(t, d.Prev[0].GetNext().Data, 10.0)
}
