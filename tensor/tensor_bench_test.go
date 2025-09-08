package tensor

import "testing"

//  1. BenchmarkNativeMul-16
//     475           2429622 ns/op          163970 B/op          4 allocs/op
func BenchmarkNativeMul(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	b.StopTimer()
	for i := 0; i < b.N; i++ {
		m1 := NewMatrix[complex128](100, 100)
		RandomTensor(m1)
		m2 := NewMatrix[complex128](100, 100)
		RandomTensor(m2)

		b.StartTimer()
		nativeMul(m1, m2)
		b.StopTimer()
	}
}
