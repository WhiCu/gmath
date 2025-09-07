package tensor

type Matrix[T Number] = Tensor[T]

func NewMatrix[T Number](x, y int) *Matrix[T] {
	return NewTensor[T](x, y)
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
			aVal := a.Data[i*a.Strides[0]+k*a.Strides[1]]
			if aVal == 0 {
				continue
			}
			for j := 0; j < p; j++ {
				bVal := b.Data[k*b.Strides[0]+j*b.Strides[1]]
				if bVal == 0 {
					continue
				}
				out.Data[i*out.Strides[0]+j*out.Strides[1]] += aVal * bVal
			}
		}
	}
	return out
}
