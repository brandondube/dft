package dft

import (
	"math"
	"math/cmplx"
)

// NewDenseMatrix produces a new (rxc) matrix backed by contiguous data.
// this function produces superior memory access patterns and prevents the rows
// of the output from being scattered in memory.
//
// data may be nil, in which case an array of zeros is returned
func NewDenseMatrixC(r, c int, data []complex128) [][]complex128 {
	if data == nil {
		data = make([]complex128, r*c)
	}
	out := make([][]complex128, r)
	for i := 0; i < r; i++ {
		// c is length of a row, offset by i rows and index up to the next one
		row := data[i*c : (i+1)*c]
		out[i] = row
	}
	return out
}

// MatVecProd computes the matrix-vector product Ax for (nxm) matrix A and (1xm)
// vector B, storing it in out
func MatVecProdCR(A [][]complex128, x []float64, out []complex128) []complex128 {
	n := len(x)
	m := len(A)
	if out == nil {
		out = make([]complex128, m)
	}
	for i := 0; i < m; i++ {
		tmp := complex(0.0, 0.0)
		for j := 0; j < n; j++ {
			tmp += A[i][j] * complex(x[j], 0)
		}
		out[i] = tmp
	}
	return out
}

func VecAbs(in []complex128, out []float64) []float64 {
	if out == nil {
		out = make([]float64, len(in))
	}
	for i := 0; i < len(in); i++ {
		out[i] = cmplx.Abs(in[i])
	}
	return out
}

// MatrixDFTExecutor performs Discrete Fourier Transforms (DFTs) using a
// matrix-vector product.  The type is not thread-safe.
//
// MatrixDFTExecutor.RunR() performs a DFT, returning the result.
//
// no shifts are required of the input or the output.
//
// The return of MatrixDFTExecutor.RunR() is a view into pre-allocated
// memory.  The caller may mutate the array, but should understand the array
// will be overwritten on the next RunR() call.
//
// It is typical to examine the magnitude or phase of the DFT result;
// these operations naturally "copy" the result.  If the complex value is to be
// used, the caller should copy the data into their own slice between RunR() calls.
//
// Recall that the Fourier Transform is the decomposition of a signal into
// a sine/cosine basis set.  For orthogonal bases, the coefficients
// are simply the dot product of the mode and the data.
//
// in other words, the real part of the DFT at a particular frequency is
// sx := sin(2*pi*x*f) // sine x
// cx := cos(2*pi*x*f) // cos x
// csf := dot(sx, data) // coefficient of sine x
// ccf := dot(cx, data) //                cos  x
//
// Recall also Euler's identify, exp(i*x) = sin(c) + i*cos(x);
// we can perform both dot products simultaneously using complex numbers
//
// Finally, recall that a matrix multiply is simply a row-wise dot product.
//
// Given all these details, the DFT can be expressed as a matrix multiply
// between a transformation matrix and the data.
//
// This matrix can be pre-computed if the data length and frequencies of interest
// are not changing.  We are also able to compute the DFT for arbitrary frequencies
// and in particular only those frequencies which are of interest.  When
// performing spectral analysis, this often makes the matrix DFT substantially
// faster than the FFT.
//
type MatrixDFTExecutor struct {
	m   [][]complex128
	out []complex128
}

func (mft MatrixDFTExecutor) RunR(in []float64) []complex128 {
	out := MatVecProdCR(mft.m, in, mft.out)
	return out
}

func NewDFTExecutorFromFreqs(N int, dx float64, freqs []float64) MatrixDFTExecutor {
	// mat will be M x N
	M := len(freqs)

	// the creation of a vector "x" which is {0..N}*dx is implicit
	// in the below; the vector is never formed explicitly
	m := NewDenseMatrixC(M, N, nil)
	out := make([]complex128, M)
	for i := 0; i < M; i++ {
		// fi := freqs[i]
		partial_kernel := 2 * math.Pi * freqs[i]
		for j := 0; j < N; j++ {
			// the DFT kernel is -1j * 2pi * f*x
			xj := float64(j) * dx
			// some micro-optimization done above, to compute 2*pi*fi less frequently
			// kernel := complex(0, 2*math.Pi*xj*fi)
			kernel := complex(0, xj*partial_kernel)
			m[i][j] = cmplx.Exp(kernel)
		}
	}

	return MatrixDFTExecutor{m: m, out: out}
}
