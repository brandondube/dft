package dft_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/brandondube/dft"
)

func floatApproxEqual(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}

func TestTritonalDFTMatchesNumpyImpl(t *testing.T) {
	dt := 1. / 44_100 // redbook mono
	N := 44_100       // 1 second window
	y := make([]float64, N)
	freqsIn := []float64{1000, 1005, 1010}
	for i := 0; i < N; i++ {
		for j := 0; j < len(freqsIn); j++ {
			xj := float64(i) * dt
			y[i] += math.Sin(2 * math.Pi * freqsIn[j] * xj)
		}
	}
	freqsOut := freqsIn
	mft := dft.NewDFTExecutorFromFreqs(N, dt, freqsOut)
	oc := mft.RunR(y)
	oabs := dft.VecAbs(oc, nil)
	// half the sequence length, since half of the power is at the negative
	// frequencies (input is real, so DFT is hermetian symmetric)
	EXPECTED := 22050.
	for i, v := range oabs {
		if !floatApproxEqual(v, float64(EXPECTED), 1e-9) {
			t.Errorf("output index %d got %f expected %f", i, v, EXPECTED)
		}
	}
}

func ExampleTritonalDFT() {
	// https://open.spotify.com/track/6S5rRj7A38pxLnPtsztXKb?si=df98c48a4a604655
	dt := 1. / 44_100 // redbook mono
	N := 44_100       // 1 second window
	y := make([]float64, N)
	freqsIn := []float64{1000, 1005, 1010}
	for i := 0; i < N; i++ {
		for j := 0; j < len(freqsIn); j++ {
			xj := float64(i) * dt
			y[i] += math.Sin(2 * math.Pi * freqsIn[j] * xj)
		}
	}
	freqsOut := freqsIn
	mft := dft.NewDFTExecutorFromFreqs(N, dt, freqsOut)
	oc := mft.RunR(y)
	oabs := dft.VecAbs(oc, nil)
	fmt.Println("Freqs:        ", freqsOut)
	fmt.Println("Fourier Coefs:", oabs)
}

func BenchmarkDFT(b *testing.B) {
	dt := 1. / 44_100 // redbook mono
	N := 44_100       // 1 second window
	y := make([]float64, N)
	freqsIn := []float64{1000, 1005, 1010}
	for i := 0; i < N; i++ {
		for j := 0; j < len(freqsIn); j++ {
			xj := float64(i) * dt
			y[i] += math.Sin(2 * math.Pi * freqsIn[j] * xj)
		}
	}
	// 9 tones
	freqsOut := []float64{900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100}
	mft := dft.NewDFTExecutorFromFreqs(N, dt, freqsOut)
	oabs := make([]float64, len(freqsOut))
	for i := 0; i < b.N; i++ {
		oc := mft.RunR(y)
		dft.VecAbs(oc, oabs)
	}
}

// func BenchmarkFFT(b *testing.B) {
// 	dt := 1. / 44_100 // redbook mono
// 	N := 44_100       // 1 second window
// 	y := make([]float64, N)
// 	freqsIn := []float64{1000, 1005, 1010}
// 	for i := 0; i < N; i++ {
// 		for j := 0; j < len(freqsIn); j++ {
// 			xj := float64(i) * dt
// 			y[i] += math.Sin(2 * math.Pi * freqsIn[j] * xj)
// 		}
// 	}
// 	fft := fourier.NewFFT(len(y))
// 	oc := make([]complex128, len(y)/2+1)
// 	oa := make([]float64, len(oc))
// 	for i := 0; i < b.N; i++ {
// 		fft.Coefficients(oc, y)
// 		dft.VecAbs(oc, oa)
// 	}
// }
