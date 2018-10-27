static char help[] = "We solve the  PFC equation in 2D rectangular domain with\n\
the SAV method\n\n";

#include <petscksp.h>
#include <petscdmda.h>
#include <petscdm.h>
#include <petsctime.h>
#include <math.h>
#include <time.h>

typedef struct {
	PetscReal      	dt,curT,endT;
  	PetscInt 		tsmax, tsstep,InterP;
  	PetscReal 		fnorm0, fnorm_ratio, norm_tol;
  	PetscReal		tmin,tmax;
  	PetscReal		alpha,Zalpha;
} tsCtx;

typedef struct {
	tsCtx		*ts;
	KSP			ksp;
	DM			da;
	Vec			x, xold;
	Vec         B, C, X, Y, b;
	Vec			x_ex, x_sum;
	Mat			A;
	
    PassiveReal Length;                          /* test problem parameter */
    PetscInt    curN;
    PetscScalar eps, beta, gamma, c0;
	PetscScalar R;
	PetscScalar stept[20000], Tenergy[20000];    /*about energy*/
    PetscScalar	SE1, SE, SE1_np1_2;
    PetscScalar	SBxphi, SBxphiP;                 /*temp variable*/
} AppCtx;

PetscErrorCode FormInitial(DM, AppCtx*);
PetscErrorCode Update(void *ptr);
PetscErrorCode SetKSP(void *ptr);
PetscErrorCode CalB(void *ptr);
PetscErrorCode CalC(void *ptr);
PetscErrorCode Calnewx(void *ptr);
PetscErrorCode Getdt(void *ptr);
PetscErrorCode phi_ex(void *ptr);

PetscErrorCode DataSaveASCII(Vec, char*);
PetscErrorCode DataLoadASCII(Vec, char*);
PetscErrorCode GetFilename(PetscInt, char*);
PetscScalar    max(PetscScalar, PetscScalar);
PetscScalar    min(PetscScalar, PetscScalar);

PetscErrorCode OutputInfoAndConfig(int, char**);

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
    par01 = 	+ ((-1.0*user->beta)/h2 + (-8.0)/h4)*pardt*dt;
    par11 = 	+  ((0.0*user->beta)/h2 +  (2.0)/h4)*pardt*dt;
    par02 = 	+  ((0.0*user->beta)/h2 +  (1.0)/h4)*pardt*dt;
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

#undef __FUNCT__
#define __FUNCT__ "DataSaveASCII"

PetscErrorCode DataSaveASCII(Vec x, char *filename){
    PetscErrorCode ierr;
    PetscViewer    dataviewer;

    PetscFunctionBegin;

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,&dataviewer); CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(dataviewer,PETSC_VIEWER_ASCII_SYMMODU); CHKERRQ(ierr);
    ierr = VecView(x,dataviewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&dataviewer); CHKERRQ(ierr);
 
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DataLoadASCII"
PetscErrorCode DataLoadASCII(Vec x, char *filename){
    PetscErrorCode ierr;
    PetscViewer    dataviewer;

    PetscFunctionBegin;

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,&dataviewer); CHKERRQ(ierr);
    ierr = VecLoad(x,dataviewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&dataviewer); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "GetFilename"
PetscErrorCode GetFilename(PetscInt i, char *filename){
    PetscInt i1,i2,i3,i4;
    
    PetscFunctionBegin;
    i1 = i%10; i/=10;
    i2 = i%10; i/=10;
    i3 = i%10; i/=10;
    i4 = i%10;
   /* filename[9] = '0' + i4;
    filename[10] = '0' + i3;
    filename[11] = '0' + i2;
    filename[12] = '0' + i1;*/
 /* filename[21] = '0' + i4;
    filename[22] = '0' + i3;
    filename[23] = '0' + i2;
    filename[24] = '0' + i1;*/

    filename[8] = '0' + i4;
    filename[9] = '0' + i3;
    filename[10] = '0' + i2;
    filename[11] = '0' + i1;
    PetscFunctionReturn(0);
}

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

#undef __FUNCT__
#define __FUNCT__ "OutputInfoAndConfig"
PetscErrorCode OutputInfoAndConfig(int argc,char **argv){
    PetscInt       i, rank;
    MPI_Comm       comm;
    FILE           *fp;
    PetscErrorCode ierr;
    char fileTE[PETSC_MAX_PATH_LEN] = "InfoAndConfig";
    
    comm = PETSC_COMM_WORLD;
    MPI_Comm_rank(comm,&rank);
    
    if (!rank) {
        ierr = PetscFOpen(comm, fileTE, "w", &fp);CHKERRQ(ierr);
        /*-------date and time in UTC/GMT+08:00--------------------------*/
		struct tm *t;            //tm结构指针
        time_t now = time(0);    //声明time_t类型变量
        time(&now);              //获取系统日期和时间
        t = localtime(&now);     //获取当地日期和时间

		PetscFPrintf(comm, fp, "%d/%d/%d ", t->tm_mon + 1, t->tm_mday, t->tm_year + 1900);
        PetscFPrintf(comm, fp, "%d:%d:%d UTC/GMT+08:00\n\n", t->tm_hour, t->tm_min, t->tm_sec);
        /*-------argv----------------------------------------------------*/
        for(i = 1;i < argc; ++i){
        	switch(argv[i][0]){
        		case '-':
        			PetscFPrintf(comm, fp, " \n%s ", argv[i]);
        			break;
        		default:
        			PetscFPrintf(comm, fp, "  %s ", argv[i]);
        			break;
			}
		}
        PetscFClose(PETSC_COMM_SELF,fp);
    }
    else{
    	PetscFunctionReturn(1);
	}
    PetscFunctionReturn(0);
}
