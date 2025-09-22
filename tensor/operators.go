package tensor

func (t *Tensor[T]) Mul(other *Tensor[T]) (err error) {
	if t.size != other.size && len(t.Shape) != len(other.Shape) {
		return ErrShapeMismatch
	}

	//TODO: добавить другие размерности
	switch {
	case len(t.Shape) == 2:
		out, err := MatMul(NewMatrixFromTenzor(t), NewMatrixFromTenzor(other))
		if err != nil {
			return err
		}
		t.Data = out.Data
		return nil
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
