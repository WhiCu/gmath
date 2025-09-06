package tensor

type Matrix[T Number] = Tensor[T]

func NewMatrix[T Number](x, y int) *Matrix[T] {
	return NewTensor[T](x, y)
}

// func (t *Matrix[T]) Transpose() *Matrix[T] {
// 	return Equal(t, other)
// }
