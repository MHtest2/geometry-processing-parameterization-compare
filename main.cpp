#include "tutte.h"
#include <igl/read_triangle_mesh.h>
#include <igl/per_vertex_normals.h>
#include <igl/viewer/Viewer.h>
#include <Eigen/Core>
#include <string>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
  // Load input meshes
  Eigen::MatrixXd V, U_tutte_e, U_tutte_g, U_tutte_t, U;
  Eigen::MatrixXi F;
  igl::read_triangle_mesh(
    (argc>1?argv[1]:"../shared/data/beetle.obj"),V,F);
  igl::viewer::Viewer viewer;
  std::cout<<R"(
[space]  Toggle whether displaying 3D surface or 2D parameterization
C,c      Toggle checkerboard
g        Switch parameterization to Tutte embedding (Graph Laplacian)
e        [DEFAULT] Switch parameterization to Tutte embedding (Edge-weighted Graph Laplacian)
t        Switch parameterization to Tutte embedding (Cotangent Laplacian)
)";

  tutte(V, F, 0, U_tutte_g);
  tutte(V, F, 1, U_tutte_e);
  tutte(V, F, 2, U_tutte_t);

  // Fit parameterization in unit sphere
  const auto normalize = [](Eigen::MatrixXd &U)
  {
    U.rowwise() -= U.colwise().mean().eval();
    U.array() /= 
      (U.colwise().maxCoeff() - U.colwise().minCoeff()).maxCoeff()/2.0;
  };
  normalize(V);
  normalize(U_tutte_g);
  normalize(U_tutte_e);
  normalize(U_tutte_t);

  bool plot_parameterization = false;
  const auto & update = [&]()
  {
    if(plot_parameterization)
    {
      // Viewer wants 3D coordinates, so pad UVs with column of zeros
      viewer.data.set_vertices(
        (Eigen::MatrixXd(V.rows(),3)<<
         U.col(0),Eigen::VectorXd::Zero(V.rows()),U.col(1)).finished());
    }else
    {
      viewer.data.set_vertices(V);
    }
    viewer.data.compute_normals();
    viewer.data.set_uv(U*10);
  };
  viewer.callback_key_pressed = 
    [&](igl::viewer::Viewer &, unsigned int key, int)
  {
    switch(key)
    {
      case ' ':
        plot_parameterization ^= 1;
        break;
      case 'e':
        U = U_tutte_e;
        break;
      case 'g':
        U = U_tutte_g;
        break;
      case 't':
        U = U_tutte_t;
        break;
      case 'C':
      case 'c':
        viewer.core.show_texture ^= 1;
        break;
      default:
        return false;
    }
    update();
    return true;
  };

  U = U_tutte_e;
  viewer.data.set_mesh(V,F);
  Eigen::MatrixXd N;
  igl::per_vertex_normals(V,F,N);
  viewer.data.set_colors(N.array()*0.5+0.5);
  update();
  viewer.core.show_texture = true;
  viewer.core.show_lines = false;
  viewer.launch();

  return EXIT_SUCCESS;
}
