#ifndef LINALG_HPP
#define LINALG_HPP

#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <Eigen/Dense>

// Klasa bazowa Tensor
class Tensor {
public:
    virtual ~Tensor() = default; // Wirtualny destruktor
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Klasa z wektorem
template<typename T>
class Vector : public Tensor {
private:
    std::vector<T> data_;
    size_t rows_;

public:
    // 1. Konstruktor domyślny
    Vector(size_t size) : data_(size, T()), rows_(size){};

    // 2. Konstruktor z danymi
    Vector(const std::vector<T>& data) : data_(data), rows_(data.size()){}

    // 3. Konstruktor kopiujący
    Vector(const Vector& other) : data_(other.data_), rows_(other.rows_){}

    // 4. Konstruktor przenoszący
    Vector(Vector&& other) noexcept : data_(std::move(other.data_)), rows_(std::move(other.rows_)) {}

    // 5. Operator przypisania kopiujący
    Vector& operator=(const Vector& other) {
        if (this != &other) {
            data_ = other.data_;
            rows_ = other.rows_;
        }
        return *this;
    }

    // 6. Operator przypisania przenoszący
    Vector& operator=(Vector&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            rows_ = std::move(other.rows_);
        }
        return *this;
    }

    // Dostęp do elementów
    T& operator[](size_t i) { 
        if(i>=rows_)
            throw std::out_of_range("Indeks wykracza poza rozmiar wektora");
        return data_[i];}

    const T& operator[](size_t i) const { 
        if(i>=rows_)
            throw std::out_of_range("Indeks wykracza poza rozmiar wektora");
        return data_[i]; }

    // Getter rozmiaru wektora
    size_t size() const { return rows_; }

    // Getter zasobów wektora
    std::vector<T> data() const { return data_; }

    // Norma kwadratowa wektora
    T norm() {
        return std::sqrt(std::inner_product(data_.begin(), data_.end(), data_.begin(), T(0)));
    }

    // Suma elementów wektora
    T sum(){
        T suma = std::accumulate(data_.begin(), data_.end(), T(0));
        return suma;
    }

    // Średnia wartość elementu wektora
    T mean(){
        return sum()/static_cast<T>(data_.size());
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Klasa z macierzą
template<typename T>
class Matrix : public Tensor {
private:
    size_t rows_, cols_;
    std::vector<T> data_;

public:
    // 1. Konstruktor domyślny
    Matrix(size_t rows = 0, size_t cols = 0)
    : rows_(rows), cols_(cols), data_(rows*cols, T()) {}

    // 2. Konstruktor kopiujący
    Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}

    // 3. Konstruktor przenoszący
    Matrix(Matrix&& other) noexcept
        : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_)) {
        other.rows_ = 0;
        other.cols_ = 0;
    }

    // 4. Operator przypisania kopiujący
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = other.data_;
        }
        return *this;
    }

    // 5. Operator przypisania przenoszący
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = std::move(other.data_);
            other.rows_ = 0;
            other.cols_ = 0;
        }
        return *this;
    }

    // Dostęp do elementów
    T& at (size_t i, size_t j) {
        if (i >= rows_ || j >= cols_)
            throw std::out_of_range("Indeks wykracza poza rozmiar macierzy");
        return data_[i * cols_ + j];
    }

    const T& at(size_t i, size_t j) const {
        if (i >= rows_ || j >= cols_)
            throw std::out_of_range("Indeks wykracza poza rozmiar macierzy");
        return data_[i * cols_ + j];
    }

    // Getter liczby wierszy
    size_t rows() const { return rows_; }

    // Getter liczby kolumn
    size_t cols() const { return cols_; }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Klasa będąca implementacją rozkładu LU nieosobliwej macierzy kwadratowej
template <typename T>
class LU {
private:
    Matrix<T> L; // Macierz dolnotrójkątna
    Matrix<T> U; // Macierz górnotrójkątna

public:
    // Konstruktor - rozkład LU metodą Doolittle'a
    LU(const Matrix<T>& m_in) {
        size_t n = m_in.rows();
        if (m_in.rows() != m_in.cols()) {
            throw std::invalid_argument("Macierz wejściowa musi być kwadratowa");
        }

        // Inicjalizacja macierzy
        L = eye<T>(n);           // Jednostkowa macierz dolnotrójkątna
        U = Matrix<T>(n, n);     // Pusta macierz górnotrójkątna

        for (size_t k = 0; k < n; ++k) {
            for (size_t j = k; j < n; ++j) {
                T sum = 0;
                for (size_t p = 0; p < k; ++p) {
                    sum += L.at(k, p) * U.at(p, j);
                }
                U.at(k, j) = m_in.at(k, j) - sum;
            }

            for (size_t i = k + 1; i < n; ++i) {
                T sum = 0;
                for (size_t p = 0; p < k; ++p) {
                    sum += L.at(i, p) * U.at(p, k);
                }
                if (U.at(k, k) == 0) {
                    throw std::runtime_error("Rozkład LU wymaga niezerowych elementów na przekątnej.");
                }
                L.at(i, k) = (m_in.at(i, k) - sum) / U.at(k, k);
            }
        }
    }

    // Domyślny destruktor
    ~LU() = default;

    // Wyznacznik macierzy
    T det() const {
        if (U.rows() != U.cols()) {
            throw std::invalid_argument("Macierz musi być kwadratowa, aby istniał jej wyznacznik.");
        }

        T determinant = 1;
        for (size_t i = 0; i < U.rows(); ++i) {
            determinant *= U.at(i, i); // Iloczyn elementów na przekątnej macierzy U
        }
        return determinant;
    }

    // Gettery do macierzy L i U
    const Matrix<T>& getL() const { return L; }
    const Matrix<T>& getU() const { return U; }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Norma kwadratowa wektora
template<typename T>
T norm(const Vector<T> &v_in){
    T norm = 0;
    for(size_t i = 0; i<v_in.size(); ++i){
        norm+= v_in[i]*v_in[i];
    }
    return sqrt(norm);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
Matrix<T> transpose(const Matrix<T> &m_in){

    Matrix<T> result(m_in.cols(), m_in.rows());

    for(size_t i = 0; i<m_in.rows(); ++i){
        for(size_t j = 0; j<m_in.cols(); ++j){
            result.at(i,j) = m_in.at(j,i);
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Metoda budująca macierz jednostkową 
template<typename T>
Matrix<T> eye(unsigned int a){

    // Tworzenie macierzy jednostkowej
    Matrix<T> result(a, a);
    for (unsigned int i = 0; i < a; ++i) {
        result.at(i, i) = 1; // Przypisanie jedynek na diagonali
    }
    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Metoda budująca macierz diagonalna
template<typename T>
Matrix<T> diag(const Vector<T> &v_in){

    // Tworzenie macierzy jednostkowej
    Matrix<T> result(v_in.size(), v_in.size());
    for (unsigned int i = 0; i < v_in.size(); ++i) {
        result.at(i, i) = v_in[i]; // Przepisanie wektora na diagonalę
    }
    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Metoda odwracająca macierz
template<typename T>
Matrix<T> inv(const Matrix<T> &m_in) {
    if (m_in.rows() != m_in.cols()) {
        throw std::invalid_argument("Macierz musi być kwadratowa, aby istniala jej odwrotnosc");
    }

    // Konwersja do macierzy z Eigen
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix(m_in.rows(), m_in.cols());
    for (size_t i = 0; i < m_in.rows(); ++i) {
        for (size_t j = 0; j < m_in.cols(); ++j) {
            eigenMatrix(i, j) = m_in.at(i, j);
        }
    }

    // Wyznaczenie odwrotności
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigenInv = eigenMatrix.inverse();

    // Konwersja z macierzy z Eigen do macierzy z biblioteki
    Matrix<T> result(m_in.rows(), m_in.cols());
    for (size_t i = 0; i < m_in.rows(); ++i) {
        for (size_t j = 0; j < m_in.cols(); ++j) {
            result.at(i, j) = eigenInv(i, j);
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Metoda zwracająca wektor wartości własnych macierzy
template<typename T>
std::vector<T> eigs(const Matrix<T>& m_in) {

    if (m_in.rows() != m_in.cols()) {
        throw std::invalid_argument("Macierz musi być kwadratowa, aby obliczyć wartości własne");
    }

    // Konwersja do macierzy Eigen
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix(m_in.rows(), m_in.cols());
    for (size_t i = 0; i < m_in.rows(); ++i) {
        for (size_t j = 0; j < m_in.cols(); ++j) {
            eigenMatrix(i, j) = m_in.at(i, j);
        }
    }

    // Obliczanie wartości własnych
    Eigen::EigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> solver(eigenMatrix);
    Eigen::VectorXcd eigenValues = solver.eigenvalues(); 

    // Sprawdzenie, czy wartości własne są rzeczywiste
    std::vector<T> result;
    for (int i = 0; i < eigenValues.size(); ++i) {
        if (std::abs(eigenValues[i].imag()) < 1e-9) { // Tolerancja na część urojoną
            result.push_back(eigenValues[i].real());
        } else {
            throw std::runtime_error("Macierz posiada wartości własne zespolone");
        }
    }

    // Sortowanie wartości własnych malejąco
    std::sort(result.begin(), result.end(), [](T a, T b){return a>b;});

    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Ślad macierzy
template<typename T>
T trace(const Matrix<T> &m_in){
    if (m_in.rows() != m_in.cols()) {
        throw std::invalid_argument("Macierz musi byc kwadratowa aby istnial jej slad");
    }
    T tr = 0;
    for(size_t i = 0; i<m_in.rows(); ++i){
        tr+=m_in.at(i,i);
    }
    return tr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Dodawanie macierzy
template<typename T>
Matrix<T> operator+(const Matrix<T>& m1, const Matrix<T>& m2) {
    if (m1.rows() != m2.rows() || m1.cols() != m2.cols()) {
        throw std::invalid_argument("Rozmiary macierzy nie pozwalaja na dodawanie");
    }
    Matrix<T> result(m1.rows(), m1.cols());
    for (size_t i = 0; i < m1.rows(); ++i) {
        for (size_t j = 0; j < m1.cols(); ++j) {
            result.at(i, j) = m1.at(i, j) + m2.at(i, j);
        }
    }
    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Mnożenie macierzy
template<typename T>
Matrix<T> operator*(const Matrix<T>& m1, const Matrix<T>& m2) {
    if (m1.cols() != m2.rows()) {
        throw std::invalid_argument("Rozmiary macierzy nie pozwalaja na mnozenie");
    }
    Matrix<T> result(m1.rows(), m2.cols());
    for (size_t i = 0; i < m1.rows(); ++i) {
        for (size_t j = 0; j < m2.cols(); ++j) {
            T sum = T();
            for(size_t k = 0; k < m1.cols(); ++k){
                sum+= m1.at(i, k) * m2.at(k,j);
            }
            result.at(i,j) = sum;   
        }
    }
    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Mnożenie macierzy przez wektor
template<typename T>
Vector<T> operator*(const Matrix<T>& m, const Vector<T>& v) {
    if (m.cols() != v.size()) {
        throw std::invalid_argument("Rozmiar macierzy i rozmiar wektora nie pozwalaja na mnozenie");
    }
    Vector<T> result(m.rows());
    for (size_t i = 0; i < m.rows(); ++i) {
        T sum = T();
        for (size_t j = 0; j < m.cols(); ++j) {
            sum+= m.at(i,j)*v[j];
        }
        result[i] = sum;
    }
    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Mnożenie macierzy przez skalar
template<typename T>
Matrix<T> operator*(const T &s, const Matrix<T>& m) {
    Matrix<T> result(m.rows(), m.cols());
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            result.at(i,j) = m.at(i,j) * s;
        }
    }
    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Mnożenie wektora przez skalar
template<typename T>
Vector<T> operator*(const T &s, const Vector<T>& v) {

    Vector<T> result(v);
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] *=s;
    }
    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Dodawanie wektorów
template<typename T>
Vector<T> operator+(const Vector<T>& v1, const Vector<T>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Rozmiary wektorow nie pozwalaja na dodawanie");
    }
    Vector<T> result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Iloczyn skalarny
template<typename T>
T dot(const Vector<T> &v1, const Vector<T> &v2){
        if (v1.size() != v2.size()) {
        throw std::invalid_argument("Rozmiary wektorow nie pozwalaja na obliczenie iloczynu skalarnego");
    }

        T result = 0;
        for(size_t i = 0; i<v1.size(); ++i){
            result+=v1[i] * v2[i];
        }

    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Iloczyn wektorowy
template<typename T>
Vector<T> cross(const Vector<T> &v1, const Vector<T> &v2){
        if (v1.size() != 3 || v2.size() != 3) {
        throw std::invalid_argument("Rozmiary wektorow nie pozwalaja na obliczenie iloczynu wektorowego");
    }

        Vector<T> out(3);
        out[0] = v1[1]*v2[2] - v1[2]*v2[1];
        out[1] = -v1[0]*v2[2] + v1[2]*v2[0];
        out[2] = v1[0]*v2[1] - v1[1]*v2[0];

    return out;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Operator drukowania macierzy
template<typename T>
std::ostream& operator<<(std::ostream& out, const Matrix<T>& m) {
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            out << m.at(i, j) << "\t";
        }
        out << std::endl;
    }
    return out;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Operator drukowania wektora
template<typename T>
std::ostream& operator<<(std::ostream& out, const Vector<T>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        out << v[i] << std::endl;
    }
    out << std::endl;
    return out;
}

#endif // LINALG_HPP