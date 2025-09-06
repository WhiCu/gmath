package tensor

import (
	"slices"

	"golang.org/x/exp/constraints"
)

type Number interface {
	constraints.Integer | constraints.Float | constraints.Complex
}

type Tensor[T Number] struct {
	Shape   []int
	size    int
	Strides []int
	Data    []T
}

func (t *Tensor[T]) checkSize(size []int) bool {
	return len(size) != len(t.Shape)
}

func NewTensor[T Number](shape ...int) *Tensor[T] {
	size := 1
	for _, d := range shape {
		size *= d
	}

	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return &Tensor[T]{
		Shape:   shape,
		size:    size,
		Strides: strides,
		Data:    make([]T, size),
	}
}

// AT || SET

func (t *Tensor[T]) At(idxs ...int) T {
	if t.checkSize(idxs) {
		panic("wrong number of indices")
	}

	offset := 0
	for i, idx := range idxs {
		if idx < 0 || idx >= t.Shape[i] {
			panic("index out of range")
		}
		offset += idx * t.Strides[i]
	}
	return t.Data[offset]
}

func (t *Tensor[T]) Set(v T, idxs ...int) error {
	if t.checkSize(idxs) {
		panic("wrong number of indices")
	}
	offset := 0
	for i, idx := range idxs {
		if idx < 0 || idx >= t.Shape[i] {
			panic("index out of range")
		}
		offset += idx * t.Strides[i]
	}
	t.Data[offset] = v
	return nil
}

// ADD || SUB || MUL || DIV

func (t *Tensor[T]) Add(other *Tensor[T]) *Tensor[T] {
	return Add(t, other)
}

func Add[T Number](a, b *Tensor[T]) *Tensor[T] {
	if !Equal(a, b) {
		panic("shape mismatch in Add")
	}
	out := NewTensor[T](a.Shape...)
	for i := range a.size {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	return out
}

func (t *Tensor[T]) Sub(other *Tensor[T]) *Tensor[T] {
	return Sub(t, other)
}

func Sub[T Number](a, b *Tensor[T]) *Tensor[T] {
	if !Equal(a, b) {
		panic("shape mismatch in Sub")
	}
	out := NewTensor[T](a.Shape...)
	for i := range a.size {
		out.Data[i] = a.Data[i] - b.Data[i]
	}
	return out
}

func (t *Tensor[T]) MulElementwise(other *Tensor[T]) *Tensor[T] {
	return Mul(t, other)
}

func Mul[T Number](a, b *Tensor[T]) *Tensor[T] {
	if !Equal(a, b) {
		panic("shape mismatch in Mul")
	}
	out := NewTensor[T](a.Shape...)
	for i := range a.size {
		out.Data[i] = a.Data[i] * b.Data[i]
	}
	return out
}

func (t *Tensor[T]) Scale(c T) *Tensor[T] {
	return Scale(t, c)
}

func Scale[T Number](a *Tensor[T], c T) *Tensor[T] {
	out := NewTensor[T](a.Shape...)
	for i := range a.Data {
		out.Data[i] = a.Data[i] * c
	}
	return out
}

// transpose

func (t *Tensor[T]) T() *Tensor[T] {
	order := make([]int, len(t.Shape))
	for i := range order {
		order[i] = len(t.Shape) - 1 - i
	}
	return t.Transpose(order...)
}

func (t *Tensor[T]) Transpose(order ...int) *Tensor[T] {
	return Transpose(t, order...)
}

func Transpose[T Number](t *Tensor[T], order ...int) *Tensor[T] {
	if len(order) == 0 {
		return t.T()
	}
	if len(order) != len(t.Shape) {
		panic("transpose: order must have the same length as the shape")
	}

	used := make(map[int]bool, len(order))
	for _, axis := range order {
		if axis < 0 || axis >= len(t.Shape) {
			panic("transpose: invalid axis in order")
		}
		if used[axis] {
			panic("transpose: duplicate axis in order")
		}
		used[axis] = true
	}

	newShape := make([]int, len(t.Shape))
	newStrides := make([]int, len(t.Strides))

	for newAxis, oldAxis := range order {
		newShape[newAxis] = t.Shape[oldAxis]
		newStrides[newAxis] = t.Strides[oldAxis]
	}

	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}

	if newSize != len(t.Data) {
		panic("transpose: internal error: size mismatch after transpose")
	}

	return &Tensor[T]{
		Shape:   newShape,
		size:    t.size,
		Strides: newStrides,
		Data:    slices.Clone(t.Data),
	}
}

// equal
func (t *Tensor[T]) Equal(other *Tensor[T]) bool {
	return Equal(t, other)
}

func Equal[T Number](a, b *Tensor[T]) bool {
	if a.size != b.size {
		return false
	}
	for i := 0; i < len(a.Shape); i++ {
		if a.Shape[i] != b.Shape[i] {
			return false
		}
	}
	return true
}
