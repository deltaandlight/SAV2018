#include <petscksp.h>
#include <petscdmda.h>
#include <petscdm.h>
#include <petsctime.h>
#include <math.h>
#include <time.h>
#include "def.h"


/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitial"
/*
 FormInitial - Forms initial values.
 
 Output Parameter:
 user.Xpre - vector
 */
PetscErrorCode FormInitial(DM da,AppCtx *user)
{
    PetscInt       i, j, Mx, My, xs, ys, xm, ym, use_random=0;
    PetscErrorCode ierr;
    PetscReal      L, hx, hy;
    PetscScalar    **x;
    PetscFunctionBeginUser;
    ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, 
	    PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, 
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);CHKERRQ(ierr);
	if (use_random)
	{
	    PetscRandom rctx;
	    PetscReal sum, mean;
        PetscRandomCreate(PETSC_COMM_WORLD, &rctx);
	    PetscRandomSetFromOptions(rctx);
	    PetscRandomSetInterval(rctx, 0.01, 0.13);
	    VecSetRandom(user->x, rctx);
        VecSum(user->x, &sum);
	    mean = sum/(double)(Mx*My);
	    VecScale(user->x, 0.07/mean);
	}
	else
	{
	    PetscReal lx, ly;
	    L  = user->Length;
        hx = L / (PetscReal)Mx;
        hy = L / (PetscReal)My;
        
	    VecSet(user->x,0);
        ierr = DMDAVecGetArray(da, user->x, &x);CHKERRQ(ierr);
        ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);CHKERRQ(ierr);
            
        for (j = ys; j < ys + ym ; ++j) {
			for (i = xs; i < xs + xm; ++i) {
				lx = i*hx;
				ly = j*hy;
		        x[j][i] = 0.05*sin(lx)*sin(ly);
			}
	    }
        ierr = DMDAVecRestoreArray(da, user->x, &x);CHKERRQ(ierr);
	}    
    PetscFunctionReturn(0);
}

