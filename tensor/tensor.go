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

func (t *Tensor[T]) Copy() *Tensor[T] {
	return &Tensor[T]{
		Shape:   slices.Clone(t.Shape),
		size:    t.size,
		Strides: slices.Clone(t.Strides),
		Data:    slices.Clone(t.Data),
		zero:    t.zero,
	}
}

func (t *Tensor[T]) SameShape(other *Tensor[T]) bool {
	return SameShape(t, other)
}

func (t *Tensor[T]) indexOffset(idxs []int) (int, error) {
	return indexOffset(t, idxs)
}

// // AT || SET

func (t *Tensor[T]) At(idxs ...int) (T, error) {
	offset, err := t.indexOffset(idxs)
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
	offset, err := t.indexOffset(idxs)
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

func (t *Tensor[T]) Equal(other *Tensor[T]) bool {
	return Equal(t, other)
}
