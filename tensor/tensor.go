package tensor

import (
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
	zero    T
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

func (t *Tensor[T]) At(idxs ...int) (T, error) {
	offset, err := indexOffset(t, idxs)
	if err != nil {
		return t.zero, err
	}
	return t.Data[offset], nil
}

func (t *Tensor[T]) MustAt(idxs ...int) T {
	v, err := t.At(idxs...)
	Must(err)
	return v
}

func (t *Tensor[T]) Set(v T, idxs ...int) error {
	offset, err := indexOffset(t, idxs)
	if err != nil {
		return err
	}
	t.Data[offset] = v
	return nil
}

func (t *Tensor[T]) MustSet(v T, idxs ...int) error {
	err := t.Set(v, idxs...)
	Must(err)
	return nil
}

// ADD || SUB || MUL || DIV || SCALE || TRANSPOSE

func (t *Tensor[T]) Add(other *Tensor[T]) (*Tensor[T], error) {
	return Add(t, other)
}

func (t *Tensor[T]) MustAdd(other *Tensor[T]) *Tensor[T] {
	out, err := Add(t, other)
	Must(err)
	return out
}

func (t *Tensor[T]) Sub(other *Tensor[T]) (*Tensor[T], error) {
	return Sub(t, other)
}

func (t *Tensor[T]) MustSub(other *Tensor[T]) *Tensor[T] {
	out, err := Sub(t, other)
	Must(err)
	return out
}

func (t *Tensor[T]) ElemMul(other *Tensor[T]) (*Tensor[T], error) {
	return ElemMul(t, other)
}

func (t *Tensor[T]) MustElemMul(other *Tensor[T]) *Tensor[T] {
	out, err := ElemMul(t, other)
	Must(err)
	return out
}

func (t *Tensor[T]) Div(other *Tensor[T]) (*Tensor[T], error) {
	return Div(t, other)
}

func (t *Tensor[T]) MustDiv(other *Tensor[T]) *Tensor[T] {
	out, err := Div(t, other)
	Must(err)
	return out
}

func (t *Tensor[T]) Scale(c T) *Tensor[T] {
	return Scale(t, c)
}

func (t *Tensor[T]) T() *Tensor[T] {
	return t.MustTranspose()
}

func (t *Tensor[T]) Transpose(order ...int) (*Tensor[T], error) {
	return Transpose(t, order...)
}

func (t *Tensor[T]) MustTranspose(order ...int) *Tensor[T] {
	out, err := Transpose(t, order...)
	Must(err)
	return out
}

func (t *Tensor[T]) Equal(other *Tensor[T]) bool {
	return Equal(t, other)
}
