package engine

import (
	"fmt"

	"github.com/tmc/dot"
)

func trace(root *Value) (map[*Value]struct{}, map[*Value][]*Value) {
	nodes := make(map[*Value]struct{})
	edges := make(map[*Value][]*Value)
	var build func(*Value)
	build = func(v *Value) {
		if _, exists := nodes[v]; !exists {
			nodes[v] = struct{}{}
			for _, child := range v.prev {
				edges[v] = append(edges[v], child)
				build(child)
			}
		}
	}
	build(root)
	fmt.Println(nodes)
	fmt.Println(edges)
	return nodes, edges
}

func DrawDot(root *Value, filename string) string {
	nodes, edges := trace(root)
	g := dot.NewGraph("G")
	g.SetType(dot.DIGRAPH)
	g.Set("rankdir", "LR")

	for n := range nodes {
		nodeID := fmt.Sprintf("%p", n)
		node, _ := g.AddNode(dot.NewNode(nodeID))
		node.Set("shape", "record")
		node.Set("label", fmt.Sprintf("{ %s | data %.4f | grad %.4f }", n.Label, n.Data, n.Grad))

		if n.op != "" {
			opNodeID := fmt.Sprintf("%p%s", n, n.op)
			opNode, _ := g.AddNode(dot.NewNode(opNodeID))
			opNode.Set("label", string(n.op))
			g.AddEdge(dot.NewEdge(opNode, node))
		}
	}

	for parent, children := range edges {
		for _, child := range children {
			childOpNodeID := fmt.Sprintf("%p%s", parent, parent.op)
			g.AddEdge(dot.NewEdge(dot.NewNode(fmt.Sprintf("%p", child)), dot.NewNode(childOpNodeID)))
		}
	}

	err := g.ToPNG(filename)
	if err != nil {
		return err.Error()
	}

	return g.String()
}
