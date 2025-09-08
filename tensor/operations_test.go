package tensor

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func expectError(t *testing.T, fn func() error) {
	t.Helper()
	So(fn, ShouldNotBeNil)
}

func expectSuccess[T Number](t *testing.T, fn func() (*Tensor[T], error)) *Tensor[T] {
	t.Helper()
	cfg, err := fn()
	So(err, ShouldBeNil)
	return cfg
}

func TestTensorOperations(t *testing.T) {
	Convey("Given two tensors with same shape", t, func() {
		a := NewTensor[int](2, 3)
		b := NewTensor[int](2, 3)

		// Fill data for testing
		for i := range a.Data {
			a.Data[i] = i + 1
			b.Data[i] = i + 10
		}

		Convey("SameShape should return true", func() {
			So(a.SameShape(b), ShouldBeTrue)
		})

		Convey("Element-wise addition should work", func() {
			c := expectSuccess(t, func() (*Tensor[int], error) { return Add(a, b) })
			for i := range c.Data {
				So(c.Data[i], ShouldEqual, a.Data[i]+b.Data[i])
			}
		})

		Convey("Element-wise subtraction should work", func() {
			c := expectSuccess(t, func() (*Tensor[int], error) { return Sub(a, b) })
			for i := range c.Data {
				So(c.Data[i], ShouldEqual, a.Data[i]-b.Data[i])
			}
		})

		Convey("Element-wise multiplication should work", func() {
			c := expectSuccess(t, func() (*Tensor[int], error) { return ElemMul(a, b) })
			for i := range c.Data {
				So(c.Data[i], ShouldEqual, a.Data[i]*b.Data[i])
			}
		})

		Convey("Element-wise division should work", func() {
			for i := range b.Data {
				if b.Data[i] == 0 {
					b.Data[i] = 1
				}
			}
			c := expectSuccess(t, func() (*Tensor[int], error) { return Div(a, b) })
			for i := range c.Data {
				So(c.Data[i], ShouldEqual, a.Data[i]/b.Data[i])
			}
		})

		Convey("Scaling works", func() {
			out := Scale(a, 3)
			for i := range a.Data {
				So(out.Data[i], ShouldEqual, a.Data[i]*3)
			}
		})

		Convey("Transposing works", func() {
			t2, err := Transpose(a)
			So(err, ShouldBeNil)
			So(t2.Shape, ShouldResemble, []int{3, 2})
			So(t2.Data, ShouldResemble, a.Data) // data order is cloned
		})
	})

	Convey("Given tensors with different shapes", t, func() {
		a := NewTensor[int](2, 3)
		b := NewTensor[int](3, 2)

		Convey("SameShape should return false", func() {
			So(a.SameShape(b), ShouldBeFalse)
		})

		Convey("Element-wise operations return error", func() {
			expectError(t, func() error {
				_, err := Add(a, b)
				return err
			})
			expectError(t, func() error {
				_, err := Sub(a, b)
				return err
			})
			expectError(t, func() error {
				_, err := ElemMul(a, b)
				return err
			})
			expectError(t, func() error {
				_, err := Div(a, b)
				return err
			})
		})
	})

	Convey("indexOffset returns correct offsets", t, func() {
		t1 := NewTensor[int](2, 3)
		offset, err := t1.indexOffset([]int{1, 2})
		So(err, ShouldBeNil)
		So(offset, ShouldEqual, 5)

		_, err = t1.indexOffset([]int{2, 0}) // out of range
		So(err, ShouldNotBeNil)

		_, err = t1.indexOffset([]int{0}) // wrong number of indices
		So(err, ShouldNotBeNil)
	})

	Convey("RandomTensor and RandomTensorN fill tensor data", t, func() {
		t1 := NewTensor[int](2, 3)
		RandomTensor(t1)
		for _, v := range t1.Data {
			So(v, ShouldNotBeNil)
		}

		RandomTensorN(t1, 5)
		for _, v := range t1.Data {
			So(v >= 0 && v < 5, ShouldBeTrue)
		}
	})
}
