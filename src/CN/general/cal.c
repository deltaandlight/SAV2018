#include <petscksp.h>
#include <petscdmda.h>
#include <petscdm.h>
#include <petsctime.h>
#include <math.h>
#include <time.h>
#include "def.h"

#undef __FUNCT__
#define __FUNCT__ "CalB"
/*
 * 
 */
PetscErrorCode CalB(void *ptr){
    AppCtx         	*user = (AppCtx*)ptr;
    PetscErrorCode 	ierr;
    PetscInt       	i, j, mx, my, xs, ys, xm, ym, xe, ye;
    PetscScalar		eps, beta, h, h2;
    PetscReal		rt_np1_2, rt_n;
	PetscReal		Lapphi;
	PetscReal		GE1, GE1_np1_2, GE, GBxphi;
    Vec				localx;
	
    ierr = DMDAGetInfo(user->da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
    ierr = DMDAGetCorners(user->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
	
    h    = user->Length/mx;
    h2   = h*h;
    eps  = user->eps;
    beta = user->beta;
    xe   = xs + xm;
    ye   = ys + ym;
    GE   = 0;
    GE1  = 0;
    GE1_np1_2 = 0;
    GBxphi    = 0;
  	
    /*----point to x,B,----------------------------------------------*/
    PetscScalar **phi, **phi_ex, **aB;/*phi=local_x*/
 
    ierr = DMGetLocalVector(user->da, &localx);CHKERRQ(ierr); 
    ierr = DMGlobalToLocalBegin(user->da, user->x, INSERT_VALUES, localx);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user->da, user->x, INSERT_VALUES, localx);CHKERRQ(ierr);	
    
    ierr = DMDAVecGetArray(user->da, localx, &phi);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da, user->x_ex, &phi_ex);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da, user->B, &aB);

    for (j = ys; j < ye; ++j) {
        for (i = xs; i < xe; ++i) {
            rt_np1_2 = (phi_ex[j][i]*phi_ex[j][i] - 1.0 - beta);
            rt_n     = (phi[j][i]*phi[j][i]       - 1.0 - beta); 
			Lapphi = ( phi[j+1][i] 
			         + phi[j-1][i] 
					 + phi[j][i+1] 
					 + phi[j][i-1] 
					 - 4.0*phi[j][i] )/h2;
			
			aB[j][i] = rt_np1_2 * phi_ex[j][i]/(eps*eps);	                /*the B is U*/
			
            GE1_np1_2 += h2*rt_np1_2*rt_np1_2/(eps*eps*4.0);	                /*local E1n+1/2*/
            GE1       += h2*rt_n*rt_n/(eps*eps*4.0);	                        /*local E1*/		
			GE        += h2*rt_n*rt_n/(eps*eps*4.0)
			+ h2*phi[j][i]*(phi[j][i]*beta/(eps*eps) - Lapphi)/2.0;    /*local E*/
            GBxphi    += h2*phi[j][i]*aB[j][i];                     /*local B*phi*/
        }
    }
    ierr = MPI_Allreduce(&GE1_np1_2, &user->SE1_np1_2, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);	
 	ierr = MPI_Allreduce(&GE1, &user->SE1, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
 	ierr = MPI_Allreduce(&GE, &user->SE, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
 	ierr = MPI_Allreduce(&GBxphi, &user->SBxphi, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
 	
  	user->R	= sqrt(user->SE1 + user->c0);
 	user->Tenergy[user->curN] = user->SE;
 	ierr = PetscPrintf(PETSC_COMM_WORLD,"R = %g, E = %g\n", user->R, user->SE);CHKERRQ(ierr);
 	user->SBxphi = user->SBxphi/sqrt(user->SE1_np1_2 + user->c0);
 			
 	/*----这里才得到真正的B!!!------------------------------*/	
 	ierr = DMDAVecRestoreArray(user->da, localx, &phi);CHKERRQ(ierr);
 	ierr = DMDAVecRestoreArray(user->da, user->B, &aB);CHKERRQ(ierr);
 	ierr = DMDAVecRestoreArray(user->da, user->x_ex, &phi_ex);CHKERRQ(ierr);
 	ierr = DMRestoreLocalVector(user->da, &localx);
 	ierr = VecScale(user->B, 1./sqrt(user->SE1_np1_2 + user->c0));CHKERRQ(ierr);
    
	PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "CalC"
/*
 * 
 */
PetscErrorCode CalC(void *ptr){
    AppCtx         	*user = (AppCtx*)ptr;
    tsCtx          	*ts = user->ts;
    PetscErrorCode 	ierr;
    PetscInt       	i,j, mx,my, xs,ys, xm,ym, xe,ye;
    PetscReal		h, h2;
	PetscReal       pardt;
    PetscReal		Lgamma, Ggamma;
    Vec				localB, LapB, localX;
	
    PetscFunctionBeginUser;
	
    ierr = DMDAGetInfo(user->da, 0, &mx, &my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
    ierr = DMDAGetCorners(user->da, &xs, &ys, NULL, &xm, &ym, NULL);CHKERRQ(ierr);
	
    h  = user->Length/mx;
    h2 = h * h;
    xe = xs + xm;
    ye = ys + ym;
    
    switch(user->curN){
        case 0:
            pardt = 1.0;
            break;
        default:
            pardt = 1.0/2.0;
	}
  	
    /*----point to x,B,C----------------------------------------------*/
    PetscScalar **phi_sum, **alocalB, **aC, **aLapB, **alocalX;/*phi=local_x*/ 
    
    ierr = DMGetGlobalVector(user->da, &LapB);CHKERRQ(ierr);
    ierr = DMGetLocalVector(user->da, &localB);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(user->da, user->B, INSERT_VALUES, localB);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user->da, user->B, INSERT_VALUES, localB);CHKERRQ(ierr);	
	
    ierr = DMDAVecGetArray(user->da, LapB, &aLapB);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da, localB, &alocalB);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da, user->C, &aC);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da, user->x_sum, &phi_sum);CHKERRQ(ierr);

    for (j = ys; j < ye; ++j) {
        for (i = xs; i < xe; ++i) {     
            aLapB[j][i] = 
			( alocalB[j+1][i] 
			+ alocalB[j-1][i] 
			+ alocalB[j][i+1] 
			+ alocalB[j][i-1]
			- 4.0*alocalB[j][i] )/h2;
			
            aC[j][i] = phi_sum[j][i] + ts->dt*aLapB[j][i]*user->R 
			+ pardt*ts->dt*aLapB[j][i]*(-1.0/2.0*user->SBxphi);
        }
    }
    
    ierr = DMDAVecRestoreArray(user->da, LapB, &aLapB);CHKERRQ(ierr);
    ierr = KSPSolve(user->ksp, LapB, user->X);CHKERRQ(ierr);
    
	ierr = DMGetLocalVector(user->da, &localX);CHKERRQ(ierr);
	ierr = DMGlobalToLocalBegin(user->da, user->X, INSERT_VALUES, localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user->da, user->X, INSERT_VALUES, localX);CHKERRQ(ierr);	
	ierr = DMDAVecGetArray(user->da, localX, &alocalX);CHKERRQ(ierr);
	
	for (j = ys; j < ye; ++j) {
        for (i = xs; i < xe; ++i) {
            Lgamma =  alocalB[j][i]*alocalX[j][i];/*这里没加负号*/
            Ggamma += Lgamma*h2;
        } 
    }	
    /*----计算 b_n*phi_(n+1)----------------------------*/ 
    ierr = MPI_Allreduce(&Ggamma, &user->gamma, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);
    ierr = DMDAVecRestoreArray(user->da, localX, &alocalX);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(user->da, &localX);
    ierr = VecDestroy(&LapB);CHKERRQ(ierr);
    
 	ierr = DMDAVecRestoreArray(user->da, localB, &alocalB);CHKERRQ(ierr);
 	ierr = DMDAVecRestoreArray(user->da, user->C, &aC);CHKERRQ(ierr);
 	ierr = DMDAVecRestoreArray(user->da, user->x_sum, &phi_sum);CHKERRQ(ierr);
 	ierr = DMRestoreLocalVector(user->da, &localB);
 	
    ierr = KSPSolve(user->ksp, user->C, user->Y);CHKERRQ(ierr);	/*得到的是一个中间参数X*/
		   
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Calnewx"
/*
 * 
 */
PetscErrorCode Calnewx(void *ptr){
  	AppCtx         	*user = (AppCtx*)ptr;
  	tsCtx          	*ts = user->ts;
  	PetscErrorCode 	ierr;
	PetscInt       	i, j, mx, my, xs, ys, xm, ym, xe, ye;
	PetscReal		h, h2, pardt;
	PetscReal		LBxphiP, GBxphiP;
	Vec				localB, localY;
	
  	PetscFunctionBeginUser;
	
  	ierr = DMDAGetInfo(user->da,0, &mx, &my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  	ierr = DMDAGetCorners(user->da, &xs, &ys, NULL, &xm, &ym, NULL);CHKERRQ(ierr);
	
	h  = user->Length/mx;
	h2 = h * h;
	xe = xs + xm;
  	ye = ys + ym;
  	
  	switch(user->curN){
  		case 0:
  			pardt = 1.0;
  			break;
  		default:
  			pardt = 1.0/2.0;
	}
  	
    /*----point to x,B,C----------------------------------------------*/
    PetscScalar **alocalB, **alocalY;/*phi=local_x*/ 
     
    ierr = DMGetLocalVector(user->da, &localY);CHKERRQ(ierr);
	ierr = VecDuplicate(localY, &localB);CHKERRQ(ierr);
	
    ierr = DMGlobalToLocalBegin(user->da, user->B, INSERT_VALUES, localB);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user->da, user->B, INSERT_VALUES, localB);CHKERRQ(ierr);	
    ierr = DMGlobalToLocalBegin(user->da, user->Y, INSERT_VALUES, localY);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user->da, user->Y, INSERT_VALUES, localY);CHKERRQ(ierr);	
	
    ierr = DMDAVecGetArray(user->da, localB, &alocalB);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da, localY, &alocalY);CHKERRQ(ierr);
	 
	GBxphiP = 0.0;    /*初始化 b_n*phi_(n+1) */ 
    for (j = ys; j < ye; ++j) {
        for (i = xs; i < xe; ++i) {
            LBxphiP =  alocalB[j][i]*alocalY[j][i]/(1 - ts->dt*pardt*user->gamma/2.0);
            GBxphiP += LBxphiP*h2;
        } 
    }	
    /*----计算 b_n*phi_(n+1)----------------------------*/ 
    ierr = MPI_Allreduce(&GBxphiP, &user->SBxphiP, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);
    		
    /*----计算最后的方程------------------*/ 
	VecAXPY(user->Y, ts->dt*pardt*user->SBxphiP/2.0, user->X);

    ierr = DMDAVecRestoreArray(user->da, localB, &alocalB);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da, localY, &alocalY);CHKERRQ(ierr);
 	ierr = DMRestoreLocalVector(user->da, &localB);
 	ierr = DMRestoreLocalVector(user->da, &localY);
 	
 	ierr = VecCopy(user->x, user->xold);
    ierr = VecCopy(user->Y, user->x);		
    /*---------------------------------------------------------------------*/
	PetscFunctionReturn(0);
}

