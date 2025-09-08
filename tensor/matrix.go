package tensor

type Matrix[T Number] = Tensor[T]

func NewMatrix[T Number](x, y int) *Matrix[T] {
	return NewTensor[T](x, y)
}

func (t *Tensor[T]) MatMul(other *Matrix[T]) error {
	if t.Shape[1] != other.Shape[0] {
		return ErrShapeMismatch
	}
	return t.nativeMul(other)
}

func (t *Tensor[T]) nativeMul(other *Matrix[T]) error {
	panic("unimplemented")
}

func MatMul[T Number](a, b *Matrix[T]) (out *Matrix[T], err error) {
	if a.Shape[1] != b.Shape[0] {
		return nil, ErrShapeMismatch
	}
	out = nativeMul(a, b)
	return out, nil
}

func nativeMul[T Number](a, b *Matrix[T]) *Matrix[T] {
	m, n, p := a.Shape[0], a.Shape[1], b.Shape[1]
	out := NewMatrix[T](m, p)

	for i := 0; i < m; i++ {
		for k := 0; k < n; k++ {
			aVal := a.MustAt(i, k)
			if aVal == 0 {
				continue
			}
			for j := 0; j < p; j++ {
				bVal := b.MustAt(k, j)
				if bVal == 0 {
					continue
				}
				out.Data[i*out.Strides[0]+j*out.Strides[1]] += aVal * bVal
			}
		}
	}
	return out
}
