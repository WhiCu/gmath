package tensor

import (
	"math/rand/v2"
	"slices"
)

func SameShape[T Number](t *Tensor[T], other *Tensor[T]) bool {
	if t.size != other.size {
		return false
	}
	if len(t.Shape) != len(other.Shape) {
		return false
	}
	for i := range t.Shape {
		if t.Shape[i] != other.Shape[i] {
			return false
		}
	}
	return true
}

func indexOffset[T Number](t *Tensor[T], idxs []int) (out int, err error) {
	const op = "indexOffset"
	defer func() {
		err = WrapIfNil(err, op)
	}()
	if len(idxs) != len(t.Shape) {
		return 0, ErrWrongNumberOfIndices
	}
	offset := 0
	for i, idx := range idxs {
		if idx < 0 || idx >= t.Shape[i] {
			return 0, ErrIndexOutOfRange
		}
		offset += idx * t.Strides[i]
	}
	return offset, nil
}

func Mul[T Number](a, b *Tensor[T]) (out *Tensor[T], err error) {
	const op = "Mul"
	defer func() {
		err = WrapIfNil(err, op)
	}()
	if a.size != b.size {
		return nil, ErrShapeMismatch
	}

	//TODO: добавить другие размерности
	switch {
	case len(a.Shape) == 2:
		return MatMul(a, b)
	default:
		return nil, ErrNotImplemented
	}
}

func elementwiseOp[T Number](a, b *Tensor[T], op func(T, T) T) (*Tensor[T], error) {
	if !SameShape(a, b) {
		return nil, ErrShapeMismatch
	}
	out := NewTensor[T](a.Shape...)
	for i := range a.Data {
		out.Data[i] = op(a.Data[i], b.Data[i])
	}
	return out, nil
}

func Add[T Number](a, b *Tensor[T]) (out *Tensor[T], err error) {
	const op = "Add"
	defer func() {
		err = WrapIfNil(err, op)
	}()

	return elementwiseOp(a, b, func(a, b T) T { return a + b })
}

func Sub[T Number](a, b *Tensor[T]) (out *Tensor[T], err error) {
	const op = "Sub"
	defer func() {
		err = WrapIfNil(err, op)
	}()

	return elementwiseOp(a, b, func(a, b T) T { return a - b })
}

func ElemMul[T Number](a, b *Tensor[T]) (out *Tensor[T], err error) {
	const op = "ElemMul"
	defer func() {
		err = WrapIfNil(err, op)
	}()

	return elementwiseOp(a, b, func(a, b T) T { return a * b })
}

func Div[T Number](a, b *Tensor[T]) (out *Tensor[T], err error) {
	const op = "Div"
	defer func() {
		err = WrapIfNil(err, op)
	}()

	return elementwiseOp(a, b, func(a, b T) T { return a / b })
}

func Scale[T Number](a *Tensor[T], c T) *Tensor[T] {
	// const op = "scale"
	out := NewTensor[T](a.Shape...)
	for i := range a.Data {
		out.Data[i] = a.Data[i] * c
	}
	return out
}

func Transpose[T Number](t *Tensor[T], order ...int) (out *Tensor[T], err error) {
	const op = "Transpose"
	defer func() {
		err = WrapIfNil(err, op)
	}()

	if len(order) == 0 {
		order = make([]int, len(t.Shape))
		for i := range order {
			order[i] = len(t.Shape) - 1 - i
		}
	}

	if len(order) != len(t.Shape) {
		return nil, ErrInvalidTransposeOrder
	}

	used := make(map[int]struct{}, len(order))
	for _, axis := range order {
		if axis < 0 || axis >= len(t.Shape) {
			return nil, ErrInvalidAxis
		}
		if _, ok := used[axis]; ok {
			return nil, ErrDuplicateAxis
		}
		used[axis] = struct{}{}
	}

	newShape := make([]int, len(t.Shape))
	newStrides := make([]int, len(t.Strides))

	for newAxis, oldAxis := range order {
		newShape[newAxis] = t.Shape[oldAxis]
		newStrides[newAxis] = t.Strides[oldAxis]
	}

	return &Tensor[T]{
		Shape:   newShape,
		size:    t.size,
		Strides: newStrides,
		Data:    slices.Clone(t.Data),
	}, nil
}

func Equal[T Number](a, b *Tensor[T]) bool {
	if SameShape(a, b) {
		return false
	}
	for i := 0; i < len(a.Shape); i++ {
		if a.Shape[i] != b.Shape[i] {
			return false
		}
	}
	return true
}

func RandomTensor[T Number](t *Tensor[T]) {
	for i := range t.Data {
		t.Data[i] = Rand[T]()
	}
}

func RandomTensorN[T Number](t *Tensor[T], n T) {
	for i := range t.Data {
		t.Data[i] = RandN(n)
	}
}

func RandN[T Number](n T) T {
	var v any = n
	var t T
	switch any(t).(type) {
	case int:
		v = rand.IntN(v.(int))
	case int8:
		v = int8(rand.IntN(int(v.(int8))))
	case int16:
		v = int16(rand.IntN(int(v.(int16))))
	case int32:
		v = int32(rand.Int32N(v.(int32)))
	case int64:
		v = int64(rand.Int64N(v.(int64)))
	case uint:
		v = uint(rand.UintN(v.(uint)))
	case uint8:
		v = uint8(rand.UintN(uint(v.(uint8))))
	case uint16:
		v = uint16(rand.UintN(uint(v.(uint8))))
	case uint32:
		v = uint32(rand.Uint32N(v.(uint32)))
	case uint64:
		v = uint64(rand.Uint64N(v.(uint64)))
	case float32:
		v = rand.Float32()
	case float64:
		v = rand.Float64()
	case complex64:
		v = complex(rand.Float32(), rand.Float32())
	case complex128:
		v = complex(rand.Float64(), rand.Float64())
	}
	return v.(T)
}

func Rand[T Number]() T {
	var v any
	var t T
	switch any(t).(type) {
	case int:
		v = rand.Int()
	case int8:
		v = int8(rand.Int())
	case int16:
		v = int16(rand.Int())
	case int32:
		v = int32(rand.Int32())
	case int64:
		v = int64(rand.Int64())
	case uint:
		v = uint(rand.Uint())
	case uint8:
		v = uint8(rand.Uint())
	case uint16:
		v = uint16(rand.Uint())
	case uint32:
		v = uint32(rand.Uint32())
	case uint64:
		v = uint64(rand.Uint64())
	case float32:
		v = rand.Float32()
	case float64:
		v = rand.Float64()
	case complex64:
		v = complex(rand.Float32(), rand.Float32())
	case complex128:
		v = complex(rand.Float64(), rand.Float64())
	}
	return v.(T)
}
