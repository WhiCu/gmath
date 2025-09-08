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

// AT || SET

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

// ADD || SUB || MUL || DIV || SCALE || TRANSPOSE

func (t *Tensor[T]) Mul(other *Tensor[T]) error {
	if t.size != other.size && len(t.Shape) != len(other.Shape) {
		return ErrShapeMismatch
	}

	//TODO: добавить другие размерности
	switch {
	case len(t.Shape) == 2:
		return t.nativeMul(other)
	default:
		return ErrNotImplemented
	}
}

func (t *Tensor[T]) ElementwiseOp(other *Tensor[T], op func(T, T) T) error {
	if !SameShape(t, other) {
		return ErrShapeMismatch
	}
	for i := range t.Data {
		t.Data[i] = op(t.Data[i], other.Data[i])
	}
	return nil
}

func (t *Tensor[T]) Add(other *Tensor[T]) error {
	return t.ElementwiseOp(other, func(a, b T) T { return a + b })
}

func (t *Tensor[T]) MustAdd(other *Tensor[T]) {
	err := t.Add(other)
	Must(err)
}

func (t *Tensor[T]) Sub(other *Tensor[T]) error {
	return t.ElementwiseOp(other, func(a, b T) T { return a - b })
}

func (t *Tensor[T]) MustSub(other *Tensor[T]) {
	err := t.Sub(other)
	Must(err)
}

func (t *Tensor[T]) ElemMul(other *Tensor[T]) error {
	return t.ElementwiseOp(other, func(a, b T) T { return a * b })
}

func (t *Tensor[T]) MustElemMul(other *Tensor[T]) {
	err := t.ElemMul(other)
	Must(err)
}

func (t *Tensor[T]) Div(other *Tensor[T]) error {
	return t.ElementwiseOp(other, func(a, b T) T { return a / b })
}

func (t *Tensor[T]) MustDiv(other *Tensor[T]) {
	err := t.Div(other)
	Must(err)
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
