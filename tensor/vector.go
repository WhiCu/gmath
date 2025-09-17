package tensor

type Vector[T Number] struct {
	*Tensor[T]
}

func NewVector[T Number](size int) *Vector[T] {
	return &Vector[T]{NewTensor[T](size)}
}
