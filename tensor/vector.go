package tensor

import (
	"fmt"
	"strings"
)

type Vector[T Number] struct {
	*Tensor[T]
}

func NewVector[T Number](size int) *Vector[T] {
	return &Vector[T]{NewTensor[T](size)}
}

func (v *Vector[T]) PrettyString() string {
	var b strings.Builder
	for i := 0; i < v.size; i++ {
		b.WriteString(fmt.Sprintf("%v ", v.MustAt(i)))
	}
	return b.String()
}
