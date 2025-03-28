#pragma once

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <random>

#include "Vector.h"

template <typename T, size_t N, size_t M>
class Matrix
{
public:
	Matrix();
	Matrix(const T& data);
	template <typename Distribution, typename Engine>
	Matrix(Distribution& dist, Engine& engine);
	Matrix(std::initializer_list<Vector<T, M>> l);
	Matrix(const Matrix& other);
	~Matrix();

	Vector<T, M>& operator[] (size_t rowIndex);
	const Vector<T, M>& operator[] (size_t rowIndex) const;
	T& operator() (size_t rowIndex, size_t colIndex);
	const T& operator() (size_t rowIndex, size_t colIndex) const;
	constexpr size_t rowSize() const;
	constexpr size_t colSize() const;
	Matrix& operator= (const Matrix& other);
	Matrix& operator= (const T& val);

	Matrix operator+ (const Matrix& other) const;
	Matrix operator- (const Matrix& other) const;
	Matrix operator-() const;
    Matrix operator*(const Matrix& other) const;
	[[nodiscard]] Matrix transpose() const;

	void modify(const std::function<T(T&)>& f);
	void fill(const T& val);
	void print() const;

	template <typename F>
	friend Matrix<T, N, M> operator*(const F& scalar, const Matrix<T, N, M>& matrix)
	{
		Matrix<T, N, M> temp;
		for (size_t i = 0; i < N; ++i)
			temp[i] = scalar * matrix.operator[](i);
		return temp;
	}

private:
	Vector<Vector<T, M>, N> m_matrix;
};

template <typename T, size_t N, size_t M>
Matrix<T, N, M>::Matrix() : m_matrix{}
{}

template <typename T, size_t N, size_t M>
Matrix<T, N, M>::Matrix(const T& data)  : m_matrix{}
{
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < M; ++j)
			m_matrix[i][j] = data;
}

template <typename T, size_t N, size_t M>
template <typename Distribution, typename Engine>
Matrix<T, N, M>::Matrix(Distribution& dist, Engine& engine)
{
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < M; ++j)
			m_matrix[i][j] = dist(engine);
}

template <typename T, size_t N, size_t M>
Matrix<T, N, M>::Matrix(std::initializer_list<Vector<T, M>> l) : m_matrix{}
{
	auto it = l.begin();
	for (size_t i = 0; i < N; ++i, ++it)
		m_matrix[i] = std::move(*it);
}

template <typename T, size_t N, size_t M>
Matrix<T, N, M>::Matrix(const Matrix& other) : m_matrix{}
{
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < M; ++j)
			m_matrix[i][j] = other.m_matrix[i][j];
}

template <typename T, size_t N, size_t M>
Matrix<T, N, M>::~Matrix() {}

template <typename T, size_t N, size_t M>
Vector <T, M> & Matrix<T, N, M>::operator[] (size_t rowIndex)
{
	assert(rowIndex < M && "Index out-of-bounds");
	return m_matrix[rowIndex];
}

template <typename T, size_t N, size_t M>
const Vector<T, M>& Matrix<T, N, M>::operator[] (size_t rowIndex) const
{
	assert(rowIndex < M && "Index out-of-bounds");
	return m_matrix[rowIndex];
}

template <typename T, size_t N, size_t M>
T& Matrix<T, N, M>::operator() (size_t rowIndex, size_t colIndex)
{
	assert(rowIndex < M && "Row index out-of-bounds");
	assert(colIndex < N && "Column index out-of-bounds");
	return m_matrix[rowIndex][colIndex];
}

template <typename T, size_t N, size_t M>
const T& Matrix<T, N, M>::operator() (size_t rowIndex, size_t colIndex) const
{
	assert(rowIndex < M && "Row index out-of-bounds");
	assert(colIndex < N && "Column index out-of-bounds");
	return m_matrix[rowIndex][colIndex];
}

template <typename T, size_t N, size_t M>
constexpr size_t Matrix<T, N, M>::rowSize() const
{
	return M;
}

template <typename T, size_t N, size_t M>
constexpr size_t Matrix<T, N, M>::colSize() const
{
	return N;
}

template <typename T, size_t N, size_t M>
Matrix<T, N, M>& Matrix<T, N, M>::operator= (const Matrix& other)
{
	if (this == &other)
		return *this;

	for (size_t i = 0; i < N; ++i)
		m_matrix[i] = other.m_matrix[i];
	return *this;
}

template <typename T, size_t N, size_t M>
Matrix<T, N, M>& Matrix<T, N, M>::operator= (const T& val)
{
	this->fill(val);
	return *this;
}

template <typename T, size_t N, size_t M>
Matrix<T, N, M> Matrix<T, N, M>::operator+ (const Matrix& other) const
{
	Matrix temp;
	for (size_t i = 0; i < N; ++i)
		temp[i] = other.operator[](i) + this->operator[](i);
	return temp;
}

template <typename T, size_t N, size_t M>
Matrix<T, N, M> Matrix<T, N, M>::operator- (const Matrix& other) const
{
	Matrix temp;
	for (size_t i = 0; i < N; ++i)
		temp[i] = other.operator[](i) + this->operator[](i);
	return temp;
}

template <typename T, size_t N, size_t M>
Matrix<T, N, M> Matrix<T, N, M>::operator-() const
{
	Matrix temp;
	for (size_t i = 0; i < N; ++i)
		temp[i] = -(this->operator[](i));
	return temp;
}

template <typename T, size_t N, size_t M>
Matrix<T,N,M> Matrix<T,N,M>::operator*(const Matrix<T,N,M>& other) const
{
    Matrix<T, N, M> C;
    // Compute the transpose of B to make accessing columns easier
    auto B_T = other.transpose(); // B_T is a Matrix<T, P, M>
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            // Dot product of row i of A and row j of B_T (which is column j of B)
            C[i][j] = m_matrix[i].dot_product(B_T[j]);
        }
    }
    return C;
}

template <typename T, size_t N, size_t M>
Matrix<T, N, M> Matrix<T, N, M>::transpose() const
{
	Matrix temp;
	for (size_t i = 0; i < N; ++i)
		for(size_t j = 0; j < M; ++j)
		temp[i][j] = m_matrix[j][i];
	return temp;
}

template <typename T, size_t N, size_t M>
void Matrix<T, N, M>::modify(const std::function<T(T&)>& f)
{
	for (size_t i = 0; i < N; ++i)
		this->operator[](i).modify(f);
}

template <typename T, size_t N, size_t M>
void Matrix<T, N, M>::fill(const T& val)
{
	for (auto& vec : m_matrix)
		vec = val;
}

template <typename T, size_t N, size_t M>
void Matrix<T, N, M>::print() const
{
	for (size_t i = 0; i < N; ++i)
	{
		for (size_t j = 0; j < M; ++j)
			std::cout << m_matrix[i][j] << " ";
		std::cout << std::endl;
	}
}