#include "lscm.h"
#include "vector_area_matrix.h"

#include "igl/massmatrix.h"
#include <igl/edge_lengths.h>

#include "cotmatrix.h"
#include "igl/cotmatrix.h"
#include "igl/repdiag.h"
#include "igl/eigs.h"
#include <Eigen/SVD>
#include <iostream>

using namespace std;

void lscm(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  Eigen::MatrixXd & U)
{
  int num_nodes = V.rows();
  U.resize(num_nodes, 2);

  Eigen::SparseMatrix<double> SignedArea;
  vector_area_matrix(F, SignedArea);

  // Obtain the Laplacian which covers the Dirichlet Energy terms
  Eigen::SparseMatrix<double>L(num_nodes, num_nodes);
  //// Old
  igl::cotmatrix(V, F, L);
  //// New
  // Eigen::MatrixXd edge_lengths;
  // igl::edge_lengths(V, F, edge_lengths);
  // cotmatrix(edge_lengths, F, L);

  Eigen::SparseMatrix<double> L_diag;
  igl::repdiag(L, 2, L_diag);
  // We factorize the 1/2 out to follow the Generalized Eigenvalue problem convention
  Eigen::SparseMatrix<double> Q = L_diag - 2 * SignedArea;

  // Mass Matrix used to enforce unit norm in our solution (this is our constraint)
  Eigen::SparseMatrix<double> M;
  //// Old
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_DEFAULT, M);
  //// New
  // M.setIdentity();

  Eigen::SparseMatrix<double> B;
  igl::repdiag(M, 2, B);

  // Borrowed from the libigl tutorial "Eigen Decomposition" section
  Eigen::MatrixXd EigenVectors;
  Eigen::VectorXd EigenValues;
  int numVectors = 3;
  igl::eigs(Q, B, numVectors, igl::EIGS_TYPE_SM, EigenVectors, EigenValues);

  cout << EigenValues << endl;

  // Eigs sorts them in descending eigenvalue order. The last eigenvalue is always 0
  int nonTrivial = numVectors - 1 - 1;
  Eigen::VectorXd FiedlerVector = EigenVectors.col(nonTrivial);

  // Set U to the non-trivial eigen vector    
  U.col(0) = FiedlerVector.head(num_nodes);
  U.col(1) = FiedlerVector.tail(num_nodes);

  cout << "min U: " << U.colwise().minCoeff() << endl;
  cout << "max U: " << U.colwise().maxCoeff() << endl;

  // Perform SVD to re-orient (PCA rotation is U)
  // Eigen::JacobiSVD<Eigen::MatrixXd> svd(U, Eigen::ComputeThinU | Eigen::ComputeThinV);
  // U = svd.matrixU();
}
