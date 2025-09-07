package tensor

import "testing"

func BenchmarkNativeMul(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	b.StopTimer()
	for i := 0; i < b.N; i++ {
		m1 := NewMatrix[complex128](100, 200)
		RandomTensor(m1)
		m2 := NewMatrix[complex128](200, 100)
		RandomTensor(m2)

		b.StartTimer()
		nativeMul(m1, m2)
		b.StopTimer()
	}
}
