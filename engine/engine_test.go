package engine_test

import (
	"fmt"
	"testing"

	"github.com/ishandhanani/micrograd-go/engine"
	"github.com/stretchr/testify/assert"
)

func Test_Add(t *testing.T) {
	v1 := &engine.Value{Data: -2, Prev: []*engine.Value{}}
	v2 := &engine.Value{Data: 3, Prev: []*engine.Value{}}
	v3 := v1.Add(v2)
	if v3.Data != 1 {
		t.Errorf("Expected %v, got %v", 1, v3.Data)
	}
	fmt.Println(v3.Prev)
	fmt.Println(v3.Prev[0].Data)
}

func Test_Multiply(t *testing.T) {
	v1 := &engine.Value{Data: 1, Prev: []*engine.Value{}}
	v2 := &engine.Value{Data: 2, Prev: []*engine.Value{}}
	v3 := v1.Add(v2)
	if v3.Data != 3 {
		t.Errorf("Expected %v, got %v", 3, v3.Data)
	}
}

func Test_VideoExample1(t *testing.T) {
	// a = 2
	// b = -3
	// c = 10
	// d._prev = {Value(-6), Value(10)}
	a := engine.Value{Data: 2.0}
	b := engine.Value{Data: -3.0}
	c := engine.Value{Data: 10.0}
	d := a.Multiply(&b).Add(&c)

	assert.Equal(t, d.Data, 4.0)
	assert.Equal(t, d.Prev[0].Data, -6.0)
}

func Test_Diagram(t *testing.T) {
	a := engine.NewValue(2.0, []*engine.Value{}, "")
	b := engine.NewValue(-3.0, []*engine.Value{}, "")
	c := engine.NewValue(10.0, []*engine.Value{}, "")
	interm := a.Multiply(b)
	d := interm.Add(c)

	str := engine.DrawDot(d)

	fmt.Println(str)
}
