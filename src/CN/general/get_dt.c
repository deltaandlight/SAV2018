#include <petscksp.h>
#include <petscdmda.h>
#include <petscdm.h>
#include <petsctime.h>
#include <math.h>
#include <time.h>
#include "def.h"

#undef __FUNCT__
#define __FUNCT__ "Getdt"
/*
 * 
 */
PetscErrorCode Getdt(void *ptr){
	AppCtx         	*user = (AppCtx*)ptr;
  	tsCtx          	*ts = user->ts;
  	PetscReal		ta,tb,norm,dG,GV,GW,dtl;
	PetscBool			flg = PETSC_TRUE;
 	if (user->curN == 0)
    {
        ts->dt = ts->tmin;
    }
    else{
    	dtl  = user->stept[user->curN - 1];
   		if (flg){
   			GV = user->SE;
     		GW = user->Tenergy[user->curN-1];
       		dG = fabs((GV-GW)/dtl);
        	ta = ts->tmin;
        	tb = ts->tmax/sqrt(1+ts->Zalpha*dG*dG);
        	ts->dt = max(ta,tb);
    	}
    	else{
    		VecAXPY(user->xold,-1.0,user->x);
        	VecNorm(user->xold,NORM_2,&norm);
        	PetscPrintf(PETSC_COMM_WORLD,"norm of two step solutions = %g\n",norm);
        	ta = ts->tmin;
        	dG = norm/dtl;
        	tb = ts->tmax/sqrt(1+ts->alpha*dG*dG);
        	ts->dt = max(ta,tb);
    	}
    }
    PetscFunctionReturn(0);
}
