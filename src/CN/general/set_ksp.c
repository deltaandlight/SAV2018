#include <petscksp.h>
#include <petscdmda.h>
#include <petscdm.h>
#include <petsctime.h>
#include <math.h>
#include <time.h>
#include "def.h"

#undef __FUNCT__
#define __FUNCT__ "SetKSP"
/*
 * 
 */
PetscErrorCode SetKSP(void *ptr){
  	AppCtx         	*user = (AppCtx*)ptr;
  	tsCtx          	*ts = user->ts;
  	PetscErrorCode 	ierr;
	PetscInt       	i,j,xe,ye;
	PetscScalar		dt,h;
	PetscReal		pardt,par00,par01,par02,par11,h2,h4;
	DMDALocalInfo	info;
	 
	
	
  	ierr = DMDAGetLocalInfo(user->da,&info);CHKERRQ(ierr);
  	
	h  = user->Length/info.mx;
  	h2 = h*h;
  	h4 = h2*h2;
  	xe = info.xs + info.xm;
  	ye = info.ys + info.ym;
  	dt = ts->dt;
  	
  	switch(user->curN){
  		case 0:
  			pardt = 1.0;
  			break;
  		default:
  			pardt = 1.0/2.0;
	}

    par00 = 1.0 +  ((4.0*user->beta)/h2 + (20.0)/h4)*pardt*dt;
    par01 = 0.0	+ ((-1.0*user->beta)/h2 + (-8.0)/h4)*pardt*dt;
    par11 = 0.0	+  ((0.0*user->beta)/h2 +  (2.0)/h4)*pardt*dt;
    par02 = 0.0	+  ((0.0*user->beta)/h2 +  (1.0)/h4)*pardt*dt;
   	/*------------------------------------------*/
	for (j = info.ys; j < ye; ++j) {
		for(i = info.xs; i<xe; ++i){
			MatStencil	row = {0};
			MatStencil  col[13] = {{0}};
			PetscScalar	v[13];
			PetscInt	ncols = 0;
			row.j = j;
			row.i = i;
			/*(0,0)点*/
			col[ncols].j = j;       col[ncols].i = i;       v[ncols++] = par00;
			/*(0,1)点*/
			col[ncols].j = j;   	col[ncols].i = i-1; 	v[ncols++] = par01;
			col[ncols].j = j;   	col[ncols].i = i+1; 	v[ncols++] = par01;
			col[ncols].j = j-1; 	col[ncols].i = i;   	v[ncols++] = par01;
			col[ncols].j = j+1; 	col[ncols].i = i;   	v[ncols++] = par01;
			/*(0,2)点*/
			col[ncols].j = j;   	col[ncols].i = i-2; 	v[ncols++] = par02;
			col[ncols].j = j;   	col[ncols].i = i+2; 	v[ncols++] = par02;
			col[ncols].j = j-2; 	col[ncols].i = i;   	v[ncols++] = par02;
			col[ncols].j = j+2; 	col[ncols].i = i;   	v[ncols++] = par02;
			/*(1,1)点*/
			col[ncols].j = j-1;   	col[ncols].i = i-1; 	v[ncols++] = par11;
			col[ncols].j = j+1;   	col[ncols].i = i+1; 	v[ncols++] = par11;
			col[ncols].j = j-1; 	col[ncols].i = i+1;   	v[ncols++] = par11;
			col[ncols].j = j+1; 	col[ncols].i = i-1;   	v[ncols++] = par11;
			ierr = MatSetValuesStencil(user->A, 1, &row, ncols, col, v, INSERT_VALUES);CHKERRQ(ierr);
		}
	}
	MatAssemblyBegin(user->A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(user->A, MAT_FINAL_ASSEMBLY);
	/*----KSP的设置----------------------------------------------*/
	ierr = KSPSetTolerances(user->ksp, 1.e-2/((info.mx+1)*(info.my+1)), 1.e-50, 
	PETSC_DEFAULT, PETSC_DEFAULT);CHKERRQ(ierr);
	ierr = KSPSetFromOptions(user->ksp);CHKERRQ(ierr);
	ierr = KSPSetOperators(user->ksp, user->A, user->A);CHKERRQ(ierr);
	PetscFunctionReturn(0);
}

