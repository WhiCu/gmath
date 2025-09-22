package tensor

import (
	"fmt"
	"strings"
)

type Matrix[T Number] struct {
	*Tensor[T]
}

func NewMatrix[T Number](x, y int) *Matrix[T] {
	return &Matrix[T]{NewTensor[T](x, y)}
}

func NewMatrixFromTenzor[T Number](t *Tensor[T]) *Matrix[T] {
	//TODO: проверку на размерность
	return &Matrix[T]{t}
}

func E[T Number](x, y int) *Matrix[T] {
	out := NewMatrix[T](x, y)
	for i := range len(out.Data) {
		out.Data[i] = 1
	}
	return out
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

func (t *Matrix[T]) SubRows(row1, row2 int) error {
	cols := t.Shape[1]
	if row1 < 0 || row1 >= t.Shape[0] || row2 < 0 || row2 >= t.Shape[0] {
		return ErrInvalidAxis
	}
	for j := 0; j < cols; j++ {
		v1 := t.MustAt(row1, j)
		v2 := t.MustAt(row2, j)
		t.MustSet(v1-v2, row1, j)
	}
	return nil
}

func (t *Matrix[T]) MustSubRows(row1, row2 int) {
	err := t.SubRows(row1, row2)
	Must(err)
}

func (t *Matrix[T]) SubCols(col1, col2 int) error {
	if len(t.Shape) != 2 {
		return ErrNotImplemented // пока только матрицы
	}
	rows := t.Shape[0]
	if col1 < 0 || col1 >= t.Shape[1] || col2 < 0 || col2 >= t.Shape[1] {
		return ErrInvalidAxis
	}
	for i := 0; i < rows; i++ {
		v1 := t.MustAt(i, col1)
		v2 := t.MustAt(i, col2)
		t.MustSet(v1-v2, i, col1)
	}
	return nil
}

func (t *Matrix[T]) MustSubCols(col1, col2 int) {
	err := t.SubCols(col1, col2)
	Must(err)
}

// SwapRows меняет местами строки row1 и row2
func (t *Matrix[T]) SwapRows(row1, row2 int) error {
	if row1 < 0 || row1 >= t.Shape[0] || row2 < 0 || row2 >= t.Shape[0] {
		return ErrInvalidAxis
	}
	if row1 == row2 {
		return nil
	}
	cols := t.Shape[1]
	for j := 0; j < cols; j++ {
		i1 := row1*t.Strides[0] + j*t.Strides[1]
		i2 := row2*t.Strides[0] + j*t.Strides[1]
		t.Data[i1], t.Data[i2] = t.Data[i2], t.Data[i1]
	}
	return nil
}

func (t *Matrix[T]) MustSwapRows(row1, row2 int) {
	err := t.SwapRows(row1, row2)
	Must(err)
}

// SwapCols меняет местами столбцы col1 и col2
func (t *Matrix[T]) SwapCols(col1, col2 int) error {
	if col1 < 0 || col1 >= t.Shape[1] || col2 < 0 || col2 >= t.Shape[1] {
		return ErrInvalidAxis
	}
	if col1 == col2 {
		return nil
	}
	rows := t.Shape[0]
	for i := 0; i < rows; i++ {
		i1 := i*t.Strides[0] + col1*t.Strides[1]
		i2 := i*t.Strides[0] + col2*t.Strides[1]
		t.Data[i1], t.Data[i2] = t.Data[i2], t.Data[i1]
	}
	return nil
}

func (t *Matrix[T]) MustSwapCols(col1, col2 int) {
	err := t.SwapCols(col1, col2)
	Must(err)
}

func (m *Matrix[T]) UpperTriangular() (*Matrix[T], error) {
	rows, cols := m.Shape[0], m.Shape[1]

	res := NewMatrix[T](rows, cols)
	copy(res.Data, m.Data)

	eps := GetEpsilon[T]()

	for k := 0; k < rows && k < cols; k++ {
		if IsLessOrEqual(Abs(res.MustAt(k, k)), eps) {
			found := false
			for i := k + 1; i < rows; i++ {
				if IsGreater(Abs(res.MustAt(i, k)), eps) {
					res.MustSwapRows(i, k)
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}

		pivot := res.MustAt(k, k)
		for i := k + 1; i < rows; i++ {
			if IsLessOrEqual(Abs(res.MustAt(i, k)), eps) {
				continue
			}
			factor := res.MustAt(i, k) / pivot
			for j := k; j < cols; j++ {
				res.Set(res.MustAt(i, j)-factor*res.MustAt(k, j), i, j)
			}
		}
	}

	return res, nil
}

func SolveGauss[T Number](a *Matrix[T], b *Vector[T]) (out *Vector[T], err error) {
	var t T
	switch any(t).(type) {
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		v, err := SolveGaussInt(any(a).(*Matrix[int]), any(b).(*Vector[int]))
		return any(v).(*Vector[T]), err
	case float32, float64, complex64, complex128:
		return SolveGaussFloat(a, b)
	default:
		return nil, ErrNotImplemented
	}
}

func SolveGaussFloat[T Number](a *Matrix[T], b *Vector[T]) (out *Vector[T], err error) {
	defer func() {
		err = WrapIfNil(err, "SolveGauss")
	}()

	rows, cols := a.Shape[0], a.Shape[1]
	if rows != b.Shape[0] {
		return nil, ErrShapeMismatch
	}

	aug := NewMatrix[T](rows, cols+1)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			aug.Set(a.MustAt(i, j), i, j)
		}
		aug.Set(b.MustAt(i), i, cols)
	}
	tri, err := aug.UpperTriangular()
	if err != nil {
		return nil, err
	}

	rankA, rankAug := RankOfMatrix(a), RankOfMatrix(tri)

	if rankA < rankAug {
		return nil, ErrNoSolution
	}
	if rankA < cols {
		return nil, ErrInfinitelyMany
	}

	eps := GetEpsilon[T]()
	x := NewVector[T](cols)
	for i := cols - 1; i >= 0; i-- {
		sum := T(0)
		for j := i + 1; j < cols; j++ {
			sum += tri.MustAt(i, j) * x.MustAt(j)
		}
		diag := tri.MustAt(i, i)
		if IsLessOrEqual(Abs(diag), eps) {
			return nil, fmt.Errorf("zero pivot on row %d", i)
		}
		val := (tri.MustAt(i, cols) - sum) / diag
		x.Set(val, i)
	}

	return x, nil
}

func RankOfMatrix[T Number](m *Matrix[T]) int {
	rows, cols := m.Shape[0], m.Shape[1]
	eps := GetEpsilon[T]()
	rank := 0

	for i := 0; i < rows; i++ {
		nonZero := false
		for j := 0; j < cols; j++ {
			if IsGreater(Abs(m.MustAt(i, j)), eps) {
				nonZero = true
				break
			}
		}
		if nonZero {
			rank++
		}
	}
	return rank
}

func (m *Matrix[T]) PrettyString() string {
	row, col := m.Shape[0], m.Shape[1]
	var b strings.Builder
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			b.WriteString(fmt.Sprintf("%v ", m.MustAt(i, j)))
		}
		b.WriteString("\n")
	}
	return b.String()
}

func SolveGaussInt(a *Matrix[int], b *Vector[int]) (*Vector[int], error) {
	rows, cols := a.Shape[0], a.Shape[1]
	if rows != b.Shape[0] {
		return nil, ErrShapeMismatch
	}

	aug := NewMatrix[int](rows, cols+1)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			aug.Set(a.MustAt(i, j), i, j)
		}
		aug.Set(b.MustAt(i), i, cols)
	}

	tri, err := aug.UpperTriangular()
	if err != nil {
		return nil, err
	}

	rankA, rankAug := RankOfMatrix(a), RankOfMatrix(tri)
	if rankA < rankAug {
		return nil, ErrNoSolution
	}
	if rankA < cols {
		return nil, ErrInfinitelyMany
	}

	x := NewVector[int](cols)

	for i := cols - 1; i >= 0; i-- {
		sum := 0
		for j := i + 1; j < cols; j++ {
			sum += tri.MustAt(i, j) * x.MustAt(j)
		}
		diag := tri.MustAt(i, i)
		rhs := tri.MustAt(i, cols) - sum

		if diag == 0 {
			return nil, fmt.Errorf("zero pivot on row %d", i)
		}

		if rhs%diag != 0 {
			return nil, ErrNoSolution
		}

		x.Set(rhs/diag, i)
	}

	return x, nil
}
