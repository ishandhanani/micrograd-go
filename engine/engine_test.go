package engine_test

import (
	"testing"

	"github.com/ishandhanani/micrograd-go/engine"
	"github.com/stretchr/testify/assert"
)

func TestAdd(t *testing.T) {
	v1 := &engine.Value{Data: 1, Prev: engine.NewTupleSet()}
	v2 := &engine.Value{Data: 2, Prev: engine.NewTupleSet()}
	v3 := v1.Add(v2)
	if v3.Data != 3 {
		t.Errorf("Expected %v, got %v", 3, v3.Data)
	}
}

func TestMultiply(t *testing.T) {
	v1 := &engine.Value{Data: 1}
	v2 := &engine.Value{Data: 2}
	v3 := v1.Multiply(v2)
	if v3.Data != 2 {
		t.Errorf("Expected %v, got %v", 2, v3.Data)
	}
}

func TestVideoExample1(t *testing.T) {
	a := engine.Value{Data: 2.0}
	b := engine.Value{Data: -3.0}
	c := engine.Value{Data: 10.0}

	res := a.Multiply(&b).Add(&c)

	assert.Equal(t, res.Data, 4.0)
}
