#include "LinAlg.hpp"
#include <iostream>

int main() {

   // Tworzenie macierzy (celowo wymiary macierzy się nie zgadzają)
    Matrix<double> A(3, 4);
    A.at(1, 1) = 10;

    Matrix<double> B(3, 3);
    B.at(1, 1) = 20;

    try {
        // Dodawanie macierzy
        Matrix<double> C = A + B;
        std::cout << "Matrix C:" << std::endl;
        std::cout << C << std::endl;
    } 
    catch (std::invalid_argument& e) {
        std::cout << "Blad podczas dodawania macierzy: " << e.what() << std::endl;
    }

    // Tworzenie wektorów
    Vector<double> v1(3);
    Vector<double> v2(3);

    v1[0] = 30;
    v1[1] = 10;
    v1[2] = 20;

    v2[1] = 5;
    v2[2] = 15;

    // Dodawanie wektorów
    Vector<double> v3 = v1 + v2;
    std::cout << "Vector v3:" << std::endl;
    std::cout << v3 << std::endl;

    // Norma wektora
    std::cout << "Norma wektora v3: " << v3.norm() << std::endl;

    // Diagonala
    Matrix<double> I(3, 3);
    I = eye<double>(3);
    std::cout << "Macierz diagonalna" << std::endl;
    std::cout << I << std::endl;

    // Mnozenie wektora przez macierz
    Vector<double> out(3);
    out = I * v3;
    std::cout << "Wektor pomnozony przez macierz" << std::endl;
    std::cout << out << std::endl;

    // Mnozenie macierzy przez macierz
    Matrix<double> M1(3,3);
    Matrix<double> M2(3,3);
    Matrix<double> M3(3,3);

    M1.at(0,0) = 2;
    M1.at(1,1) = 4;
    M1.at(2,2) = 3;
    M1.at(1,2) = 3;
    M1.at(2,1) = 3;

    M2.at(0,0) = 2;
    M2.at(1,1) = 4;
    M2.at(2,2) = 3;
    M2.at(1,2) = 3;
    M2.at(2,1) = 3;

    M3.at(0,0) = 2;
    M3.at(1,1) = 4;
    M3.at(2,2) = 3;
    M3.at(1,2) = 3;
    M3.at(2,1) = 3;

    Matrix<double> mult_out = M1*M2*M3;
    std::cout << "Mnozenie macierzy przez siebie" << std::endl;
    std::cout << M1 << "\n * \n" << M2 << "\n * \n" << M3 << "\n = \n" << mult_out << std::endl;

    // Mnożenie macierzy przez skalar
    Matrix<double> M_eye = 5.0*eye<double>(3);
    std::cout << "Wynik mnozenia macierzy eye(3) przez skalar" << std::endl;
    std::cout << M_eye << std::endl;

    // Diagonala
    Matrix<double> M_diag = diag(v1);
    std::cout << "Macierz diagonalna" << std::endl;
    std::cout << M_diag << std::endl;

    // Odwrotność
    unsigned int a = 5;
    Matrix<double> M_base(3,3);
    M_base.at(0,0) = 2;
    M_base.at(1,1) = 4;
    M_base.at(2,2) = 8;
    M_base.at(0,1) = 6;
    M_base.at(1,0) = 6;
    M_base.at(2,0) = 7;

    std::cout << "Macierz M_base" << std::endl;
    std::cout << M_base << std::endl;

    Matrix<double> M_inv = inv(M_base);
    std::cout << "Macierz odwrotna do M_base" << std::endl;
    std::cout << M_inv << std::endl;

    // Sprawdzenie odwrotności
    Matrix<double> check = M_base*M_inv;
    std::cout << "Sprawdzenie odwrotnosci (oczekujemy macierzy identycznosci)" << std::endl;
    std::cout << check << std::endl;

    // Transpozycja
    Matrix<double> M_t = transpose(M_base);
    std::cout << "Macierz M_t" << std::endl;
    std::cout << M_t << std::endl;

    // Rozkład LU
    LU<double> LU_mbase(M_base);
    Matrix<double> L = LU_mbase.getL();
    std::cout << "Macierz trojkatna L" << std::endl;
    std::cout << L << std::endl;

    Matrix<double> U = LU_mbase.getU();
    std::cout << "Macierz trojkatna U" << std::endl;
    std::cout << U << std::endl;

    Matrix<double> M_LU = L*U;
    std::cout << "Test rozkladu LU" << std::endl;
    std::cout << M_LU << " =\n" << M_base << std::endl;

    // Wyznacznik
    std::cout << "Wyznacznik macierzy M_base: " << LU_mbase.det() << std::endl;

    // Wartości własne
    std::vector<double> eigenvalues = eigs(M_base);
    
    std::cout << "Wartosci wlasne macierzy M_base" << std::endl;
    for(auto iter = eigenvalues.begin(); iter!=eigenvalues.end(); ++iter){
        std::cout << *iter << std::endl;
    }

    // Ślad macierzy
    double Tr = trace(M_base);
    std::cout << "Slad macierzy M_base: " << Tr << std::endl; 
    return 0;
}