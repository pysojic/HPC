#pragma once

#include <iostream>

// CONST ITERATOR
template <typename T, size_t N>
class VectorConstIterator
{
public:
	// The STL expects the exact traits defined below to work properly
	using iterator_category = std::random_access_iterator_tag;
	using difference_type   = ptrdiff_t;
	using value_type        = T;
	using pointer           = const T *;
	using reference         = const T &;

public:
	VectorConstIterator() : m_Ptr(nullptr), m_Idx(0) {}
	explicit VectorConstIterator(pointer ptr, size_t idx = 0) : m_Ptr(ptr), m_Idx(idx) {}

	pointer operator->() const
	{
		return m_Ptr + m_Idx;
	}

	reference operator*() const 
	{
		return *operator->();
	}

	VectorConstIterator& operator++()
	{
		++m_Idx;
		return *this;
	}

	VectorConstIterator operator++(int)
	{
		VectorConstIterator temp = *this;
		++* this;
		return temp;
	}

	VectorConstIterator& operator--() {
		--m_Idx;
		return *this;
	}

	VectorConstIterator operator--(int)
	{
		VectorConstIterator temp = *this;
		--* this;
		return temp;
	}

	VectorConstIterator& operator+=(const difference_type offset)
	{
		m_Idx += offset;
		return *this;
	}

	VectorConstIterator operator+(const difference_type offset) const
	{
		VectorConstIterator temp = *this;
		return temp += offset;
	}

	VectorConstIterator& operator-=(const difference_type offset)
	{
		return *this += -offset;
	}

	VectorConstIterator operator-(const difference_type offset) const
	{
		VectorConstIterator temp = *this;
		return temp -= offset;
	}

	difference_type operator-(const VectorConstIterator& other) const 
	{ 
		return m_Idx - other.m_Idx; 
	}

	reference operator[](const difference_type offset) const
	{
		return *(*this + offset);
	}

	bool operator==(const VectorConstIterator& _Right) const
	{
		return m_Idx == _Right.m_Idx;
	}

	bool operator!=(const VectorConstIterator& _Right) const
	{
		return !(*this == _Right);
	}

	bool operator<(const VectorConstIterator& _Right) const
	{
		return m_Idx < _Right.m_Idx;
	}

	bool operator>(const VectorConstIterator& _Right) const
	{
		return _Right < *this;
	}

	bool operator<=(const VectorConstIterator& _Right) const
	{
		return !(_Right < *this);
	}

	bool operator>=(const VectorConstIterator& _Right) const
	{
		return !(*this < _Right);
	}

private:
	pointer m_Ptr;
	size_t m_Idx;
};

// NON-CONST ITERATOR
template <typename T, size_t N>
class VectorIterator
{
public:
	// The STL expects the exact traits defined below to work properly
	using iterator_category = std::random_access_iterator_tag;
	using difference_type   = ptrdiff_t;
	using value_type        = T;
	using pointer           = T *;
	using reference         = T &;

public:
	VectorIterator() : m_Ptr(nullptr), m_Idx(0) {}
	explicit VectorIterator(pointer ptr, size_t idx = 0) : m_Ptr(ptr), m_Idx(idx) {}

	pointer operator->() const 
	{
		return m_Ptr + m_Idx;
	}

	reference operator*() const 
	{
		return *operator->();
	}

	VectorIterator& operator++() 
	{
		++m_Idx;
		return *this;
	}

	VectorIterator operator++(int) 
	{
		VectorIterator temp = *this;
		++*this;
		return temp;
	}

	VectorIterator& operator--() {
		--m_Idx;
		return *this;
	}

	VectorIterator operator--(int)
	{
		VectorIterator temp = *this;
		--*this;
		return temp;
	}

	VectorIterator& operator+=(const difference_type offset)
	{
		m_Idx += offset;
		return *this;
	}

	VectorIterator operator+(const difference_type offset) const
	{
		VectorIterator temp = *this;
		return temp += offset;
	}

	VectorIterator& operator-=(const difference_type offset)
	{
		return *this += -offset;
	}

	VectorIterator operator-(const difference_type offset) const
	{
		VectorIterator temp = *this;
		return temp -= offset;
	}

	// Calculates the distance between two iterators. Required for std::transform
	difference_type operator-(const VectorIterator& other) const
	{
		return m_Idx - other.m_Idx;
	}

	reference operator[](const difference_type offset) const
	{
		return *(*this + offset);
	}

	bool operator==(const VectorIterator& _Right) const
	{
		return m_Idx == _Right.m_Idx;
	}

	bool operator!=(const VectorIterator& _Right) const
	{
		return !(*this == _Right);
	}

	bool operator<(const VectorIterator& _Right) const 
	{
		return m_Idx < _Right.m_Idx;
	}

	bool operator>(const VectorIterator& _Right) const 
	{
		return _Right < *this;
	}

	bool operator<=(const VectorIterator& _Right) const
	{
		return !(_Right < *this);
	}

	bool operator>=(const VectorIterator& _Right) const
	{
		return !(*this < _Right);
	}

private:
	pointer m_Ptr;
	size_t m_Idx;
};