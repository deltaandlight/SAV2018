#include <petscksp.h>
#include <petscdmda.h>
#include <petscdm.h>
#include <petsctime.h>
#include <math.h>
#include <time.h>
#include "def.h"

#undef __FUNCT__
#define __FUNCT__ "phi_ex"
PetscErrorCode phi_ex(void *ptr){
  	AppCtx           *user = (AppCtx*)ptr;
    tsCtx            *ts = user->ts;
  	PetscErrorCode 	 ierr;
	PetscInt       	 i, j, mx, my, xs, ys, xm, ym, xe, ye, Ii, Jj;
	PetscReal		 h,h2,h4;
	PetscReal        LapLapphi;
	Vec				 localx,localxold;
	PetscScalar		 U; 
	PetscScalar		eps, beta;

    eps  = user->eps;
    beta = user->beta;
	
  	PetscFunctionBeginUser;
	
  	ierr = DMDAGetInfo(user->da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  	ierr = DMDAGetCorners(user->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  	
	h  = user->Length/mx;
  	h2 = h*h;
  	h4 = h2*h2;
	xe = xs + xm;
  	ye = ys + ym;
  	
    /*----point to x,B,----------------------------------------------*/
    PetscScalar **phi, **ax_ex, **alocalxold, **ax_sum;/*phi=local_x*/
 
    ierr = DMGetLocalVector(user->da,&localx);CHKERRQ(ierr); 
    ierr = VecDuplicate(localx,&localxold);CHKERRQ(ierr);
    
    ierr = DMGlobalToLocalBegin(user->da, user->x, INSERT_VALUES, localx);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user->da, user->x, INSERT_VALUES, localx);CHKERRQ(ierr);	
	ierr = DMGlobalToLocalBegin(user->da, user->xold, INSERT_VALUES, localxold);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user->da, user->xold, INSERT_VALUES, localxold);CHKERRQ(ierr);
    
    ierr = DMDAVecGetArray(user->da, localx, &phi);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da, user->x_ex, &ax_ex);
    ierr = DMDAVecGetArray(user->da, localxold, &alocalxold);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da, user->x_sum, &ax_sum);CHKERRQ(ierr);
	
	switch(user->curN){
  		case 0:
  			for (j = ys; j < ye; ++j) {
        		for (i = xs; i < xe; ++i) {
            		ax_ex[j][i]  = phi[j][i];
            		ax_sum[j][i] = phi[j][i];
        		}
    		}
  			break;
  		default:
  			for (j = ys; j < ye; ++j) {
        		for (i = xs; i < xe; ++i) {
            		//ax_ex[j][i]  = 3.0/2.0*phi[j][i] - 1.0/2.0*alocalxold[j][i];
            		Ii = i;
					Jj = j;
					U = (phi[Jj][Ii]*phi[Jj][Ii] - 1.0 - beta) * phi[Jj][Ii]/(eps*eps);
					ax_ex[j][i] = -4.0*ts->dt*U/h2/2.0;
					
					Ii = i+1;
					Jj = j;
					U = (phi[Jj][Ii]*phi[Jj][Ii] - 1.0 - beta) * phi[Jj][Ii]/(eps*eps);
					ax_ex[j][i] += ts->dt*U/h2/2.0;
					
					Ii = i-1;
					Jj = j;
					U = (phi[Jj][Ii]*phi[Jj][Ii] - 1.0 - beta) * phi[Jj][Ii]/(eps*eps);
					ax_ex[j][i] += ts->dt*U/h2/2.0;
					
					Ii = i;
					Jj = j+1;
					U = (phi[Jj][Ii]*phi[Jj][Ii] - 1.0 - beta) * phi[Jj][Ii]/(eps*eps);
					ax_ex[j][i] += ts->dt*U/h2/2.0;
					
					Ii = i;
					Jj = j-1;
					U = (phi[Jj][Ii]*phi[Jj][Ii] - 1.0 - beta) * phi[Jj][Ii]/(eps*eps);
					ax_ex[j][i] += ts->dt*U/h2/2.0;
					
            		ax_ex[j][i] += phi[j][i];
            		LapLapphi = 
			            (  1.0*phi[j+2][i]   + 1.0*phi[j-2][i]   + 1.0*phi[j][i+2]   + 1.0*phi[j][i-2]
			            +  2.0*phi[j+1][i+1] + 2.0*phi[j+1][i-1] + 2.0*phi[j-1][i-1] + 2.0*phi[j-1][i+1]
			            -  8.0*phi[j+1][i]   - 8.0*phi[j-1][i]   - 8.0*phi[j][i+1]   - 8.0*phi[j][i-1]
			            + 20.0*phi[j][i]   ) / h4;
					ax_sum[j][i] = phi[j][i] - LapLapphi*ts->dt/2.0;
        		}
    		}
	}
 	
 	ierr = DMDAVecRestoreArray(user->da, localx, &phi);CHKERRQ(ierr);
 	ierr = DMDAVecRestoreArray(user->da, user->x_ex, &ax_ex);CHKERRQ(ierr);
 	if(user->curN != 0){
 		KSPSolve(user->ksp,user->x_ex,user->x_ex);
	 }
 	ierr = DMDAVecRestoreArray(user->da, user->x_sum, &ax_sum);CHKERRQ(ierr);
 	ierr = DMRestoreLocalVector(user->da, &localx);
 	ierr = DMRestoreLocalVector(user->da, &localxold);
    
	PetscFunctionReturn(0);
}
