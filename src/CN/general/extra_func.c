#include <petscksp.h>
#include <petscdmda.h>
#include <petscdm.h>
#include <petsctime.h>
#include <math.h>
#include <time.h>
#include "def.h"

#undef __FUNCT__
#define __FUNCT__ "max"
PetscScalar max(PetscScalar a, PetscScalar b){
    if (a>b){return a;}
    else{return b;}
}

#undef __FUNCT__
#define __FUNCT__ "min"
PetscScalar min(PetscScalar a, PetscScalar b){
    if (a>b){return b;}
    else{return a;}
}
