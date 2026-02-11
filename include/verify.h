#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>

#include "utils.h"

// ============================================================================
// LU Decomposition Verification
// ============================================================================

// Set upper triangle entries (excluding diagonal entries) in column-major order
// to zero. Then, transpose to row-major order.
inline void cleanCusolverLUDecompositionResult(double *L, double *U, const int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            std::swap(L[i + j * n], L[i * n + j]);
            U[i * n + j] = L[i * n + j];
            L[i * n + j] = 0;
        }
        L[i * n + i] = 1;
    }
}

inline bool verifyLUDecomposition(double *A, double *L, double *U, const int n, bool verbose = false)
{
    auto newA = std::make_unique<double[]>(n * n);
    memset(newA.get(), 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                newA[i * n + j] += L[i * n + k] * U[k * n + j];
            }
        }
    }

    double error = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            error += fabs(A[i * n + j] - newA[i * n + j]);
        }
    }

    if (verbose)
    {
        printf("A:\n");
        printSquareMatrix(A, n);
        printf("\nnewA:\n");
        printSquareMatrix(newA.get(), n);
        printf("\nL:\n");
        printSquareMatrix(L, n);
        printf("\nU:\n");
        printSquareMatrix(U, n);
        printf("\n");
    }

    printf("error = %.6f\n", error);

    return error <= 1e-6;
}

// ============================================================================
// Cholesky Decomposition Verification
// ============================================================================

// Set upper triangle entries (excluding diagonal entries) in column-major order
// to zero. Then, transpose to row-major order.
inline void cleanCusolverCholeskyDecompositionResult(double *L, const int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            L[i + j * n] = 0;
            std::swap(L[i + j * n], L[i * n + j]);
        }
    }
}

inline bool verifyCholeskyDecomposition(double *A, double *L, const int n, bool verbose = false)
{
    auto newA = std::make_unique<double[]>(n * n);
    memset(newA.get(), 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                newA[i * n + j] += L[i * n + k] * L[k + j * n];
            }
        }
    }

    double error = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            error += fabs(A[i * n + j] - newA[i * n + j]);
        }
    }

    if (verbose)
    {
        printf("A:\n");
        printSquareMatrix(A, n);
        printf("\nnewA:\n");
        printSquareMatrix(newA.get(), n);
        printf("\nL:\n");
        printSquareMatrix(L, n);
        printf("\n");
    }

    printf("error = %.6f\n", error);

    return error <= 1e-6;
}
