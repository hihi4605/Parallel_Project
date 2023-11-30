
/************************************************************/
// Prevent multiple inclusion

#pragma once

/************************************************************/
// System includes

#include <new>
#include <algorithm>
#include <memory>

/************************************************************/

template<typename T>
class Matrix
{
public:
    using iterator = T*;
    using const_iterator = T const*;

    /**********************************************************/

    // Delete default ctor. User should always specify a size. 
    Matrix() = delete;

    /**********************************************************/

    // Initialize a square matrix of order 'order'.
Matrix(unsigned order)
    : m_order(order)
    , m_data(reinterpret_cast<T*>(new alignas(64) unsigned char[order * order * sizeof(T)]))
{
    // Note: You may need to adjust the alignment depending on your platform's requirements
    // and the alignment constraints of your data type T.
}


    /**********************************************************/

    // Copy ctor
    // Use a delegating ctor
    Matrix(Matrix const& m)
        : Matrix(m.order())
    {
        auto j = m.begin();
        auto i = begin();
        while (i != end())
        {
            *i = *j;
            ++i;
            ++j;
        }
    }

    /**********************************************************/

    // Move ctor. Default is fine. 
    Matrix(Matrix&&) = default;

    /**********************************************************/

    // Dtor. Default is fine thanks to our pal std::unique_ptr!
    ~Matrix() = default;

    /**********************************************************/

    // Copy assignment
    // Handle self assignment 
    Matrix&
        operator= (Matrix const& m)
    {
        if (this == &m)
        {
            return *this;
        }
        this = Matrix(m.order());
        for (size_t i = 0; i < order(); ++i)
        {
            for (size_t j = 0; j < order(); ++j)
                this(i, j) == m(i, j);
        }
        return *this;
    }

    /**********************************************************/

    // Move assignment. Default is fine.
    Matrix&
        operator= (Matrix&&) = default;

    /**********************************************************/

    // Return the appropriate element
    // Do NOT do bounds checking
    T&
        operator () (unsigned row, unsigned col)
    {
        return m_data.get()[row * order() + col];
    }

    /**********************************************************/

    // Return the appropriate element
    // Do NOT do bounds checking
    T const&
        operator () (unsigned row, unsigned col) const
    {
        return m_data.get()[row * order() + col];
    }

    /**********************************************************/

    // Return the order
    unsigned
        order() const
    {
        return m_order;
    }

    /**********************************************************/

    // Return pointer to first element
    iterator
        begin()
    {
        return m_data.get();
    }

    /**********************************************************/

    // Return pointer to first element
    const_iterator
        begin() const
    {
        return m_data.get();
    }

    /**********************************************************/

    // Return pointer to one beyond last element
    iterator
        end()
    {
        return begin() + ((order() * order()));
    }

    /**********************************************************/

    // Return pointer to one beyond last element
    const_iterator
        end() const
    {
        return begin() + (order() * order());
    }

    /**********************************************************/

private:
    static constexpr std::align_val_t CACHE_LINE_BYTES{ 64 };
    static constexpr auto Deleter = [](T* ptr)
        {
            operator delete[](ptr, CACHE_LINE_BYTES);
        };

private:
    std::unique_ptr<T, decltype (Deleter)> m_data;
    unsigned m_order;
};