package engine_test

import (
	"fmt"
	"testing"

	"github.com/ishandhanani/micrograd-go/engine"
	"github.com/stretchr/testify/assert"
)

func Test_Add(t *testing.T) {
	v1 := engine.NewValue(-2.0, "", []*engine.Value{})
	v2 := engine.NewValue(3.0, "", []*engine.Value{})
	v3 := v1.Add(v2)
	if v3.Data != 1 {
		t.Errorf("Expected %v, got %v", 1, v3.Data)
	}
}

func Test_Multiply(t *testing.T) {
	v1 := engine.NewValue(-2.0, "", []*engine.Value{})
	v2 := engine.NewValue(3.0, "", []*engine.Value{})
	v3 := v1.Multiply(v2)
	if v3.Data != -6 {
		t.Errorf("Expected %v, got %v", -6, v3.Data)
	}
}

func Test_VideoExample1(t *testing.T) {
	// a = 2
	// b = -3
	// c = 10
	a := engine.Value{Data: 2.0}
	b := engine.Value{Data: -3.0}
	c := engine.Value{Data: 10.0}
	d := a.Multiply(&b).Add(&c)

	assert.Equal(t, d.Data, 4.0)
}

func Test_Diagram(t *testing.T) {
	a := engine.NewValue(-2.0, "a", []*engine.Value{})
	b := engine.NewValue(3.0, "b", []*engine.Value{})
	c := engine.NewValue(10.0, "c", []*engine.Value{})
	e := a.Multiply(b).AddLabel("e")
	d := e.Add(c).AddLabel("d")

	str := engine.DrawDot(d)

	fmt.Println(str)
}
