add_library(StripePatterns
            src/Complex.cpp
            src/DenseMatrix.cpp
            src/Edge.cpp
            src/Face.cpp
            src/HalfEdge.cpp
            src/LinearContext.cpp
            src/LinearEquation.cpp
            src/LinearPolynomial.cpp
            src/LinearSystem.cpp
            src/Mesh.cpp
            src/MeshIO.cpp
            src/Quaternion.cpp
            src/Real.cpp
            src/SparseMatrix.cpp
            src/Variable.cpp
            src/Vector.cpp
            src/Vertex.cpp
        )
target_link_libraries(StripePatterns MeshFEM)
set_target_properties     (StripePatterns PROPERTIES CXX_STANDARD 14)
set_target_properties     (StripePatterns PROPERTIES CXX_STANDARD_REQUIRED ON)
target_include_directories(StripePatterns PUBLIC include)
