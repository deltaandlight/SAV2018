#include <petscksp.h>
#include <petscdmda.h>
#include <petscdm.h>
#include <petsctime.h>
#include <math.h>
#include <time.h>
#include "def.h"

#undef __FUNCT__
#define __FUNCT__ "Update"
/*
 * time stepping function
 */
PetscErrorCode Update(void *ptr){
  	AppCtx         	*user = (AppCtx*)ptr;
  	PetscErrorCode 	ierr;
  	ierr = KSPCreate(PETSC_COMM_WORLD,&user->ksp);CHKERRQ(ierr);
	/*----KSP Setting--------------------------------------------------------------*/
	SetKSP(user);		
    /*-----------------------------------------------------------------------------*/ 
     /*----B,C,X,b分配空间(x已经分配空间了)----------------------------------------*/
	ierr = VecDuplicate(user->x, &user->B);CHKERRQ(ierr); 
    ierr = VecDuplicate(user->x, &user->C);CHKERRQ(ierr); 
    ierr = VecDuplicate(user->x, &user->X);CHKERRQ(ierr);
    ierr = VecDuplicate(user->x, &user->Y);CHKERRQ(ierr);
    /*----compute phi_n+1(explicit)------------------------------------------------*/
    phi_ex(user);
    /*----compute B----------------------------------------------------------------*/
	CalB(user);
	/*----compute time step--------------------------------------------------------*/
 	Getdt(user);
 	/*----compute C----------------------------------------------------------------*/
    CalC(user);
	/*----compute phi_n+1----------------------------------------------------------*/
    Calnewx(user);
	 /*----------------------------------------------------------------------------*/ 	
	ierr = KSPDestroy(&user->ksp);CHKERRQ(ierr);
  	PetscFunctionReturn(0);
}

