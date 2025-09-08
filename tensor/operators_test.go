package tensor

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// Проверяет, что функция не паникует
func expectNoPanic(fn func()) {
	So(fn, ShouldNotPanic)
}

// Проверяет, что функция не паникует и возвращает результат
func expectSuccessNoPanic[T any](t *testing.T, fn func() *T, check func(*T)) {
	t.Helper()
	var res *T
	So(func() { res = fn() }, ShouldNotPanic)
	check(res)
}

func TestTensorMethods(t *testing.T) {
	Convey("Given two tensors of the same shape", t, func() {
		a := NewTensor[int](2, 3)
		b := NewTensor[int](2, 3)
		for i := range a.Data {
			a.Data[i] = i + 1
			b.Data[i] = i + 10
		}

		Convey("Add should modify the tensor correctly", func() {
			err := a.Add(b)
			So(err, ShouldBeNil)
			for i := range a.Data {
				So(a.Data[i], ShouldEqual, (i+1)+(i+10))
			}
		})

		Convey("MustAdd should not panic and modify correctly", func() {
			expectSuccessNoPanic(t, func() *Tensor[int] {
				a.MustAdd(b)
				return a
			}, func(res *Tensor[int]) {
				for i := range res.Data {
					So(res.Data[i], ShouldEqual, (i+1)+(i+10))
				}
			})
		})

		Convey("Sub should modify the tensor correctly", func() {
			c := NewTensor[int](2, 3)
			for i := range c.Data {
				c.Data[i] = i + 5
			}
			err := c.Sub(b)
			So(err, ShouldBeNil)
			for i := range c.Data {
				So(c.Data[i], ShouldEqual, (i+5)-(i+10))
			}
		})

		Convey("ElemMul should modify the tensor correctly", func() {
			c := NewTensor[int](2, 3)
			for i := range c.Data {
				c.Data[i] = i + 1
			}
			err := c.ElemMul(b)
			So(err, ShouldBeNil)
			for i := range c.Data {
				So(c.Data[i], ShouldEqual, (i+1)*(i+10))
			}
		})

		Convey("Div should modify the tensor correctly", func() {
			c := NewTensor[int](2, 3)
			for i := range c.Data {
				c.Data[i] = (i + 10) * 2
				if b.Data[i] == 0 {
					b.Data[i] = 1
				}
			}
			err := c.Div(b)
			So(err, ShouldBeNil)
			for i := range c.Data {
				So(c.Data[i], ShouldEqual, ((i+10)*2)/(i+10))
			}
		})
	})

	Convey("Given tensors of different shapes", t, func() {
		a := NewTensor[int](2, 3)
		b := NewTensor[int](3, 2)

		Convey("Add should return ErrShapeMismatch", func() {
			err := a.Add(b)
			So(err, ShouldEqual, ErrShapeMismatch)
		})

		Convey("Sub should return ErrShapeMismatch", func() {
			err := a.Sub(b)
			So(err, ShouldEqual, ErrShapeMismatch)
		})

		Convey("ElemMul should return ErrShapeMismatch", func() {
			err := a.ElemMul(b)
			So(err, ShouldEqual, ErrShapeMismatch)
		})

		Convey("Div should return ErrShapeMismatch", func() {
			err := a.Div(b)
			So(err, ShouldEqual, ErrShapeMismatch)
		})
	})

	Convey("Testing Scale", t, func() {
		a := NewTensor[int](2, 2)
		a.Data = []int{1, 2, 3, 4}
		out := a.Scale(3)
		So(out.Data, ShouldResemble, []int{3, 6, 9, 12})
	})

	Convey("Testing Transpose", t, func() {
		a := NewTensor[int](2, 3)
		for i := range a.Data {
			a.Data[i] = i + 1
		}

		Convey("Transpose without order", func() {
			t2 := a.MustTranspose()
			So(t2.Shape, ShouldResemble, []int{3, 2})
			So(t2.Data, ShouldResemble, a.Data)
		})

		Convey("Transpose with specific order", func() {
			t2, err := a.Transpose(1, 0)
			So(err, ShouldBeNil)
			So(t2.Shape, ShouldResemble, []int{3, 2})
			So(t2.Data, ShouldResemble, a.Data)
		})
	})

	Convey("Testing Mul for 2D tensors", t, func() {
		a := NewTensor[int](2, 3)
		b := NewTensor[int](3, 2)
		for i := range a.Data {
			a.Data[i] = i + 1
		}
		for i := range b.Data {
			b.Data[i] = i + 1
		}

		err := a.Mul(b)
		So(err, ShouldBeNil)
		So(a.Shape, ShouldResemble, []int{2, 2}) // after MatMul
	})
}
