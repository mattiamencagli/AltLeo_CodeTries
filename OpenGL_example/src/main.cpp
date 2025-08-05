#include "init.hpp"

int main(int argc, char **argv){

    std::cout << "\n ---------------------------- START ---------------------------- \n" << std::endl;
    
    Initialize(&argc, argv);
    
    glutMainLoop();

    std::cout << "\n ----------------------------- END ----------------------------- \n" << std::endl;

    return 0;
}
