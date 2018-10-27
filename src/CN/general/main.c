static char help[] = "We solve the  PFC equation in 2D rectangular domain with\n\
the SAV method\n\n";

#include <petscdmda.h>
#include <petscdm.h>
#include <petsctime.h>
#include <math.h>
#include <time.h>
#include "def.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{                        
    tsCtx          ts;
  	AppCtx         user;                /* user-defined work context */
  	PetscErrorCode ierr;
  	MPI_Comm       comm;
    
    PetscBool      flg     = PETSC_FALSE;
    PetscBool      flgJ    = PETSC_FALSE;
    
    PetscLogDouble t1, t2, t3, t4;
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = PetscInitialize(&argc, &argv, (char*)0, help); if (ierr) return(1);
    PetscFunctionBeginUser;
    /*Output Info And Configuration*/
    OutputInfoAndConfig(argc, argv);
    
    comm = PETSC_COMM_WORLD;
    ierr = PetscMalloc(sizeof(AppCtx), &user);CHKERRQ(ierr);
    /*
     Create DM
     */
    ierr = DMDACreate2d(comm, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX,
	-128, -128, PETSC_DECIDE, PETSC_DECIDE, 1, 2, 0, 0, &user.da);CHKERRQ(ierr);
    ierr = DMSetFromOptions(user.da);CHKERRQ(ierr);
    ierr = DMSetUp(user.da);CHKERRQ(ierr);
    ierr = DMCreateMatrix(user.da,&user.A);CHKERRQ(ierr);
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    user.beta   = 0.0;
    user.eps    = 0.1;
    user.Length = 2.0*M_PI;
    user.c0     = 0;
    user.ts     = &ts;
    
    ts.endT     = 0.032;
    ts.curT     = 0.0;
    ts.InterP   = 10;
    ts.tmin     = 1e-4;
	ts.tmax     = 1e-4;
	ts.dt       = 1e-4;
	ts.Zalpha   = 400000;
	ts.alpha    = 4000000;
    
    ierr = PetscOptionsGetReal(NULL,"-beta", &user.beta, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,"-eps", &user.eps, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,"-c0", &user.c0, NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-Length", &user.Length, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,"-endT", &ts.endT, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,"-Tmin", &ts.tmin, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,"-Tmax", &ts.tmax, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,"-interP", &ts.InterP, NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"beta    = %g\n", (double)user.beta);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"epsilon = %g\n", (double)user.eps);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Length  = %g\n", (double)user.Length);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"endTime = %g\n", (double)ts.endT);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"interP  = %d\n", ts.InterP);CHKERRQ(ierr);
    
    ierr = DMSetApplicationContext(user.da, &user);CHKERRQ(ierr);

    /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	ierr = DMCreateGlobalVector(user.da, &user.x);CHKERRQ(ierr);
    ierr = VecDuplicate(user.x, &user.xold);CHKERRQ(ierr);
    ierr = VecDuplicate(user.x, &user.x_ex);CHKERRQ(ierr);
    ierr = VecDuplicate(user.x, &user.x_sum);CHKERRQ(ierr);
    
    flg  = PETSC_FALSE;
    flgJ = PETSC_FALSE;
    ierr = PetscOptionsGetReal(NULL,"-dt", &ts.dt, &flg);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,"-fd", &flgJ, NULL);CHKERRQ(ierr);
    
	ierr = FormInitial(user.da, &user);CHKERRQ(ierr);
	ierr = VecCopy(user.x, user.xold);CHKERRQ(ierr);
	ierr = VecCopy(user.x, user.x_ex);CHKERRQ(ierr);
	ierr = VecCopy(user.x, user.x_sum);CHKERRQ(ierr);

    char  filenameascii[PETSC_MAX_PATH_LEN] = "ASCIsol_0000";
    ierr = DataSaveASCII(user.x, filenameascii);CHKERRQ(ierr);
	
	user.curN = 0;
    if (flg){
        PetscTime(&t1);
        do{
        	PetscTime(&t3);
            Update(&user);
			PetscTime(&t4);
			
			PetscPrintf(PETSC_COMM_WORLD, "INFO:time of this step equals %f seconds.\n", t4-t3);
            ierr = PetscPrintf(comm, "Time = %g G = %g\n",
			    ts.curT+ts.dt, (double)user.Tenergy[user.curN]);CHKERRQ(ierr);
			
            user.stept[user.curN] = ts.dt;
            user.curN = user.curN +1;
            ts.curT = ts.curT + ts.dt;
            if (user.curN%ts.InterP == 0)
            {
				ierr = GetFilename(user.curN,filenameascii);CHKERRQ(ierr);
                ierr = DataSaveASCII(user.x, filenameascii);CHKERRQ(ierr);
            }
            if (user.curN > 1)
            {
                if(fabs(user.Tenergy[user.curN-1]-user.Tenergy[user.curN-2]) < 1.0e-5)
                {
                    break;
                }
            }           
        }while(ts.curT < ts.endT);
        PetscTime(&t2);
    }
    else{
        PetscTime(&t1);
        do{
        	PetscTime(&t3);
            Update(&user);
            PetscTime(&t4);
            
            user.stept[user.curN] = ts.dt;
            user.curN = user.curN +1;
            ts.curT = ts.curT + ts.dt;
            
            PetscPrintf(PETSC_COMM_WORLD, "INFO:\n time of this step equals %f seconds.\n", t4-t3);
            ierr = PetscPrintf(comm,"curN = %d, deltaT = %g, curT=%g, Tenergy=%g\n\n",
			    user.curN, (double)ts.dt,ts.curT,user.Tenergy[user.curN-1]);CHKERRQ(ierr);
			
            if (user.curN%ts.InterP==0)
            {
				ierr = GetFilename(user.curN,filenameascii);CHKERRQ(ierr);
                ierr = DataSaveASCII(user.x, filenameascii);CHKERRQ(ierr);
 
            }
        }while(ts.curT < ts.endT);
        PetscTime(&t2);
    }
    
    PetscPrintf(PETSC_COMM_WORLD,"total time equals %f seconds.\n",t2-t1);
    char fileLastuASCII[PETSC_MAX_PATH_LEN] = "ASCIsol_0000";
    ierr = GetFilename(user.curN, fileLastuASCII);CHKERRQ(ierr);
    ierr = DataSaveASCII(user.x, fileLastuASCII);CHKERRQ(ierr);

    PetscInt i,rank;
    FILE *fp;
    char fileTE[PETSC_MAX_PATH_LEN] = "BinaStepAndEnergy";
    MPI_Comm_rank(comm,&rank);
    
    if (!rank) {
        ierr = PetscFOpen(PETSC_COMM_SELF, fileTE, "w", &fp);
        for (i = 0; i < user.curN; ++i)
        {
            PetscFPrintf(PETSC_COMM_SELF, fp, "%g %g\n", user.stept[i], user.Tenergy[i]);CHKERRQ(ierr);
        }
        PetscFClose(PETSC_COMM_SELF,fp);
    }
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*ierr = KSPDestroy(&user.ksp);CHKERRQ(ierr);*/
	ierr = VecDestroy(&user.x);CHKERRQ(ierr);
    ierr = VecDestroy(&user.xold);CHKERRQ(ierr);
    ierr = VecDestroy(&user.x_ex);CHKERRQ(ierr);
    ierr = VecDestroy(&user.x_sum);CHKERRQ(ierr);
    ierr = VecDestroy(&user.B);CHKERRQ(ierr);
    ierr = VecDestroy(&user.C);CHKERRQ(ierr);
    ierr = VecDestroy(&user.X);CHKERRQ(ierr);
    ierr = VecDestroy(&user.Y);CHKERRQ(ierr);
    ierr = MatDestroy(&user.A);CHKERRQ(ierr);
    ierr = DMDestroy(&user.da);CHKERRQ(ierr);
    
    ierr = PetscFinalize();
    return 0;
}
