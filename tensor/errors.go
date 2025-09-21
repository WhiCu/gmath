package tensor

import (
	"errors"
	"fmt"
)

var (
	// Индексация
	ErrWrongNumberOfIndices = errors.New("wrong number of indices")
	ErrIndexOutOfRange      = errors.New("index out of range")

	// Операции с тензорами
	ErrShapeMismatch = errors.New("shape mismatch")
	ErrSizeMismatch  = errors.New("size mismatch")

	// Транспонирование
	ErrInvalidTransposeOrder = errors.New("order must have the same length as shape")
	ErrInvalidAxis           = errors.New("invalid axis in order")
	ErrDuplicateAxis         = errors.New("duplicate axis in order")

	// Решение СЛАУ
	ErrSingularMatrix = errors.New("singular matrix")
	ErrNoSolution     = errors.New("no solution")
	ErrInfinitelyMany = errors.New("infinitely many solutions")

	// DEV
	ErrNotImplemented = errors.New("not implemented")
)

func Wrap(err error, msg string) error {
	return fmt.Errorf("%s: %v", msg, err)
}

func WrapIfNil(err error, msg string) error {
	if err == nil {
		return nil
	}
	return Wrap(err, msg)
}

func Must(err error) {
	if err != nil {
		panic(err)
	}
}
