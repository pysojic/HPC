#pragma once

#include <functional>
#include <algorithm>
#include <string>
#include <type_traits>
#include <numeric>

#include "VectorIterator.h"

template <typename T, size_t N>
class Vector
{
public:
	using Iterator      = VectorIterator<T, N>;
	using ConstIterator = VectorConstIterator<T, N>;

public: 
	Vector();
	Vector(const T& data);
	Vector(const Vector& other);
	Vector(Vector&& other) noexcept;
	Vector(std::initializer_list<T> list);
	~Vector();

	T& operator[] (size_t index);
	const T& operator[] (size_t index) const; // Necessary to access const objects
	T& front() noexcept;
	const T& front() const noexcept;
	T& back() noexcept;
	const T& back() const noexcept;
	constexpr size_t size() const;
	Vector& operator= (const Vector& other);
	
	Vector operator+ (const Vector& other) const;
	Vector operator- (const Vector& other) const;
	Vector operator- () const;
	T dot_product (const Vector& other) const;

	void modify(const std::function<T(T&)>& f);
	void fill(const T& val);

	// Iterator-related member functions
	Iterator begin();
	Iterator end();
	ConstIterator begin() const; // Needed instead of cbegin() to use, for e.g., range-based loops on const vectors
	ConstIterator end() const;   // Indeed, ranged-based loop only accept .begin() functions
	ConstIterator cbegin() const;
	ConstIterator cend() const;

	template <typename F>
	friend Vector<T, N> operator*(const F& scalar, const Vector<T, N>& vec)
	{
		Vector temp;
		std::transform(vec.begin(), vec.end(), temp.begin(), [&scalar](const auto& a) {return scalar * a; });
		return temp;
	}

private:
	T m_arr[N];
};

template <typename T, size_t N>
Vector<T, N>::Vector() : m_arr{}
{}

template <typename T, size_t N>
Vector<T, N>::Vector(const T& data) : m_arr{}
{
	for (auto& elem : m_arr)
		elem = data;
}

template <typename T, size_t N>
Vector<T, N>::Vector(Vector&& other) noexcept
{
	for (size_t i = 0; i < other.size(); ++i)
		m_arr[i] = std::move(other.m_arr[i]);
}

template <typename T, size_t N> 
Vector<T, N>::Vector(const Vector<T,N>& vec) : m_arr{}
{
	for (size_t i = 0; i < N; ++i)
		m_arr[i] = vec.m_arr[i];
}

template <typename T, size_t N>
Vector<T, N>::Vector(std::initializer_list<T> list) : m_arr{}
{
	std::copy(list.begin(), list.end(), this->begin());
}

template <typename T, size_t N>
Vector<T, N>::~Vector()
{
}

template <typename T, size_t N>
T& Vector<T, N>::operator[] (size_t index)
{
	assert(index < this->size() && "Index out-of-bounds");
	return m_arr[index];
}

template <typename T, size_t N>
const T& Vector<T, N>::operator[] (size_t index) const
{
	assert(index < this->size() && "Index out-of-bounds");
	return m_arr[index];
}

template <typename T, size_t N>
T& Vector<T, N>::front() noexcept
{
	return m_arr[0];
}

template <typename T, size_t N>
const T& Vector<T, N>::front() const noexcept
{
	return m_arr[0];
}

template <typename T, size_t N>
T& Vector<T, N>::back() noexcept
{
	return m_arr[this->size() - 1];
}

template <typename T, size_t N>
const T& Vector<T, N>::back() const noexcept
{
	return m_arr[this->size() - 1];
}

template <typename T, size_t N>
constexpr size_t Vector<T, N>::size() const
{
	return sizeof(m_arr) / sizeof(T);
}

template <typename T, size_t N>
Vector<T, N>& Vector<T, N>::operator= (const Vector& other)
{
	if (this == &other)
		return *this;

	for (size_t i = 0; i < N; ++i)
		m_arr[i] = other.m_arr[i];
	return *this;
}

template <typename T, size_t N>
Vector<T, N> Vector<T, N>::operator+ (const Vector& other) const
{
	Vector temp;
	std::transform(this->begin(), this->end(), other.begin(), temp.begin(), [](const auto& a, const auto& b) {return a + b; });
	return temp;
}

template <typename T, size_t N>
Vector<T, N> Vector<T, N>::operator- (const Vector& other) const
{
	Vector temp;
	std::transform(this->begin(), this->end(), other.begin(), temp.begin(), [](const auto& a, const auto& b) {return a - b; });
	return temp;
}

template <typename T, size_t N>
Vector<T, N> Vector<T, N>::operator-() const
{
	Vector temp;
	std::transform(this->begin(), this->end(), temp.begin(), [](const auto& a) {return -a; });
	return temp;
}

template <typename T, size_t N>
T Vector<T, N>::dot_product(const Vector& other) const
{
	return std::inner_product(begin(), end(), other.begin(), 0);
}


template <typename T, size_t N>
void Vector<T, N>::modify(const std::function<T(T&)>& f)
{
	std::transform(this->begin(), this->end(), this->begin(), f);
}

template <typename T, size_t N>
void Vector<T, N>::fill(const T& val)
{
	for (size_t i = 0; i < N; ++i)
		m_arr[i] = val;
}

template <typename T, size_t N>
typename Vector<T,N>::Iterator Vector<T, N>::begin()
{
	return Iterator(m_arr); // Default index at 0
}

template <typename T, size_t N>
typename Vector<T, N>::Iterator Vector<T, N>::end()
{
	return Iterator(m_arr, N);
}

template <typename T, size_t N>
typename Vector<T, N>::ConstIterator Vector<T, N>::begin() const
{
	return ConstIterator(m_arr); // Default index at 0
}

template <typename T, size_t N>
typename Vector<T, N>::ConstIterator Vector<T, N>::end() const
{
	return ConstIterator(m_arr, N);
}

template <typename T, size_t N>
typename Vector<T, N>::ConstIterator Vector<T, N>::cbegin() const
{
	return begin(); // Default index at 0
}

template <typename T, size_t N>
typename Vector<T, N>::ConstIterator Vector<T, N>::cend() const
{
	return end();
}

