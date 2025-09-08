package tensor

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func expectTensorEqual[T Number](t *testing.T, a, b *Tensor[T]) {
	t.Helper()
	So(a.Shape, ShouldResemble, b.Shape)
	So(a.Data, ShouldResemble, b.Data)
}

func TestTensorBasic(t *testing.T) {
	Convey("Given a new tensor", t, func() {
		t1 := NewTensor[int](2, 3)

		Convey("It should have correct shape, size and strides", func() {
			So(t1.Shape, ShouldResemble, []int{2, 3})
			So(t1.Strides, ShouldResemble, []int{3, 1})
			So(len(t1.Data), ShouldEqual, 6)
		})

		Convey("Setting and getting values works", func() {
			err := t1.Set(42, 1, 2)
			So(err, ShouldBeNil)

			v, err := t1.At(1, 2)
			So(err, ShouldBeNil)
			So(v, ShouldEqual, 42)

			Convey("MustAt returns value without error", func() {
				v2 := t1.MustAt(1, 2)
				So(v2, ShouldEqual, 42)
			})

			Convey("MustSet works without panic", func() {
				So(func() { t1.MustSet(99, 0, 0) }, ShouldNotPanic)
				So(t1.MustAt(0, 0), ShouldEqual, 99)
			})
		})

		Convey("Accessing out of bounds returns error", func() {
			_, err := t1.At(2, 0)
			So(err, ShouldNotBeNil)

			err = t1.Set(1, 0, 3)
			So(err, ShouldNotBeNil)
		})

		Convey("Copy creates a new independent tensor", func() {
			t2 := t1.Copy()
			expectTensorEqual(t, t1, t2)

			// Changing copy does not affect original
			t2.MustSet(123, 0, 0)
			So(t1.MustAt(0, 0), ShouldNotEqual, 123)
			So(t2.MustAt(0, 0), ShouldEqual, 123)
		})

		Convey("SameShape checks correctly", func() {
			t2 := NewTensor[int](2, 3)
			t3 := NewTensor[int](3, 2)
			So(t1.SameShape(t2), ShouldBeTrue)
			So(t1.SameShape(t3), ShouldBeFalse)
		})

		Convey("Equal checks correctly", func() {
			t2 := t1.Copy()
			So(t1.Equal(t2), ShouldBeTrue)

			t3 := NewTensor[int](2, 3)
			t3.MustSet(1, 0, 0)
			So(t1.Equal(t3), ShouldBeFalse)

			t4 := NewTensor[int](3, 2)
			So(t1.Equal(t4), ShouldBeFalse)
		})
	})
}

func TestIndexOffsetErrors(t *testing.T) {
	Convey("Given a tensor", t, func() {
		t1 := NewTensor[int](2, 3)

		Convey("Providing wrong number of indices returns error", func() {
			_, err := t1.indexOffset([]int{1})
			So(err, ShouldNotBeNil)
		})

		Convey("Providing out of range index returns error", func() {
			_, err := t1.indexOffset([]int{0, 3})
			So(err, ShouldNotBeNil)
		})
	})
}
