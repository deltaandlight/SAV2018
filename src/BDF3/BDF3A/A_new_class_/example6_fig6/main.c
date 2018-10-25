static char help[] = "We solve the  equation in 2D rectangular domain with\n\
the SAV method\n\n";

#include <petscksp.h>
#include <petscdmda.h>
#include <petscdm.h>
#include <math.h>
#include <petsctime.h>

typedef struct {
	PetscReal      	dt,curT,endT;
  	PetscInt 		tsmax, tsstep,InterP;
  	PetscReal 		fnorm0, fnorm_ratio, norm_tol;
  	PetscReal		tmin,tmax,test;
  	PetscReal		alpha,Zalpha;
} tsCtx;

typedef struct {
	tsCtx		*ts;
	KSP			ksp;
	DM			da;
	Vec			x,xold,xold2,B,C,X,Y,b;
	Vec			x_ex,x_sum;
	Mat			A;
	
    PassiveReal Length;          /* test problem parameter */
    PetscInt   curN;
    PetscScalar beta,eps,R,Rold,Rold2,gamma,c0,stept[20000];
    PetscScalar	SE1,SE,SE1_nplus1_2,Tenergy[20000];/*about energy*/
    PetscScalar	SBxphi,SBxphiP;/*temp variable*/
} AppCtx;

extern PetscErrorCode FormInitial(DM,AppCtx*);

extern PetscErrorCode Update(void *ptr);
extern PetscErrorCode SetKSP(void *ptr);
extern PetscErrorCode CalB(void *ptr);
extern PetscErrorCode Getdt(void *ptr);
extern PetscErrorCode CalC(void *ptr);
extern PetscErrorCode Calnewx(void *ptr);

extern PetscErrorCode phi_ex(void *ptr);
extern PetscScalar BDFi(void *ptr,PetscInt,PetscReal*);
extern PetscScalar max(PetscScalar,PetscScalar);
extern PetscScalar min(PetscScalar,PetscScalar);
extern PetscErrorCode DataSaveBin(Vec, char*);
extern PetscErrorCode DataSaveASCII(Vec, char*);
extern PetscErrorCode DataLoadBin(Vec, char*);
extern PetscErrorCode DataSaveVtk(Vec, char*);
extern PetscErrorCode DataLoadVtk(Vec, char*);
extern PetscErrorCode GetFilename(PetscInt, char*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{                        
    tsCtx          ts;
  	AppCtx         user;                /* user-defined work context */
  	PetscErrorCode ierr;
  	MPI_Comm       comm;
    
    PetscBool      flg              = PETSC_FALSE;
    PetscBool      flgJ             = PETSC_FALSE;
    
    PetscLogDouble t1,t2,t3,t4;
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return(1);
    
    PetscFunctionBeginUser;
    comm = PETSC_COMM_WORLD;
    ierr = PetscMalloc(sizeof(AppCtx),&user);CHKERRQ(ierr);
    /*
     Create DM
     */
    ierr = DMDACreate2d(comm,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX,-4,-4,PETSC_DECIDE,PETSC_DECIDE,1,2,0,0,&user.da);CHKERRQ(ierr);
    ierr = DMSetFromOptions(user.da);CHKERRQ(ierr);
    ierr = DMSetUp(user.da);CHKERRQ(ierr);
    ierr = DMCreateMatrix(user.da,&user.A);
    /*ierr = KSPCreate(PETSC_COMM_WORLD,&user.ksp);CHKERRQ(ierr);*//*put it in SetKSP*/
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    user.beta=0.0;
    user.eps=0.1;
    user.Length = 2*M_PI;
    user.c0=0;
    
    ts.test=1;
    ts.endT = 0.032;
    ts.curT = 0.0;
    ts.InterP = 10;
    ts.tmin = 0.0001;
	ts.tmax = 0.0001;
	ts.dt = 0.0001;
	ts.Zalpha = 400000;
	ts.alpha=4000000;
    
    ierr = PetscOptionsGetReal(NULL,"-beta",&user.beta,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,"-eps",&user.eps,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,"-c0",&user.c0,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-Length",&user.Length,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,"-endT",&ts.endT,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,"-Tmin",&ts.tmin,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,"-Tmax",&ts.tmax,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,"-T_test",&ts.test,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,"-interP",&ts.InterP,NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"beta = %15.12g\n",(double)user.beta);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"epsilon = %15.12g\n",(double)user.eps);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"L = %15.12g\n",(double)user.Length);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"endTime = %g\n",(double)ts.endT);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"interP = %d\n",ts.InterP);CHKERRQ(ierr);
    
    user.ts=&ts;
    ierr = DMSetApplicationContext(user.da,&user);CHKERRQ(ierr);

    /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	ierr = DMCreateGlobalVector(user.da,&user.x);CHKERRQ(ierr);
    ierr = VecDuplicate(user.x,&user.xold);CHKERRQ(ierr);
    ierr = VecDuplicate(user.x,&user.xold2);CHKERRQ(ierr);
    
    ierr = VecDuplicate(user.x,&user.x_ex);CHKERRQ(ierr);
    ierr = VecDuplicate(user.x,&user.x_sum);CHKERRQ(ierr);
    
    flg = PETSC_FALSE;
    flgJ = PETSC_FALSE;
    ierr = PetscOptionsGetReal(NULL,"-dt",&ts.dt,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,"-fd",&flgJ,NULL);CHKERRQ(ierr);
    
	ierr = FormInitial(user.da,&user);CHKERRQ(ierr);
	ierr = VecCopy(user.x,user.xold);CHKERRQ(ierr);
	ierr = VecCopy(user.x,user.xold2);CHKERRQ(ierr);
	
	ierr = VecCopy(user.x,user.x_ex);CHKERRQ(ierr);
	ierr = VecCopy(user.x,user.x_sum);CHKERRQ(ierr);

    char  filenamevtk[PETSC_MAX_PATH_LEN] = "Datasol_0000.vts";
    char  filenamebin[PETSC_MAX_PATH_LEN] = "Binasol_0000";
    char  filenameascii[PETSC_MAX_PATH_LEN] = "ASCIsol_0000";

    //ierr = DataSaveVtk(user.x, filenamevtk);
    //ierr = DataSaveBin(user.x, filenamebin);
    ierr = DataSaveASCII(user.x, filenameascii);
	
	user.curN = 0;
    if (flg){
        PetscTime(&t1);
        do{
        	PetscTime(&t3);
            Update(&user);
			PetscTime(&t4);
			PetscPrintf(PETSC_COMM_WORLD,"INFO:time of this step equals %f seconds.\n",t4-t3);
            ierr = PetscPrintf(comm,"Time = %g G = %g\n",ts.curT+ts.dt,(double)user.Tenergy[user.curN]/user.Length/user.Length);CHKERRQ(ierr);
            user.stept[user.curN] = ts.dt;
            user.curN = user.curN +1;
            ts.curT = ts.curT + ts.dt;
            if (user.curN%ts.InterP==0)
            {
                //ierr = GetFilename(user.curN,filenamevtk);
                //ierr = DataSaveVtk(user.x, filenamevtk);
                //ierr = GetFilename(user.curN,filenamebin);
                //ierr = DataSaveBin(user.x, filenamebin);
				ierr = GetFilename(user.curN,filenameascii);
                ierr = DataSaveASCII(user.x, filenameascii);  
            }
            if (user.curN>1)
            {
                if(fabs(user.Tenergy[user.curN-1]-user.Tenergy[user.curN-2])<1.0e-5)
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
			PetscPrintf(PETSC_COMM_WORLD,"INFO:\n time of this step equals %f seconds.\n",t4-t3);
            user.stept[user.curN] = ts.dt;
            user.curN = user.curN +1;
            ts.curT = ts.curT + ts.dt;
            ierr = PetscPrintf(comm,"curN = %d, deltaT = %g, curT=%g, Tenergy=%g\n\n",user.curN,(double)ts.dt,ts.curT,user.Tenergy[user.curN-1]/user.Length/user.Length);CHKERRQ(ierr);
            if (user.curN%ts.InterP==0)
            {
                //ierr = GetFilename(user.curN,filenamevtk);
                //ierr = DataSaveVtk(user.x, filenamevtk);
                //ierr = GetFilename(user.curN,filenamebin);
                //ierr = DataSaveBin(user.x, filenamebin);
				ierr = GetFilename(user.curN,filenameascii);
                ierr = DataSaveASCII(user.x, filenameascii);
 
            }
        }while(ts.curT < ts.endT);
        PetscTime(&t2);
    }
    
    
    PetscPrintf(PETSC_COMM_WORLD,"total time equals %f seconds.\n",t2-t1);
    char fileLastu[PETSC_MAX_PATH_LEN] = "Binasol_0000";
    ierr = GetFilename(user.curN,fileLastu);
    ierr = DataSaveBin(user.x, fileLastu);
    char fileLastuASCII[PETSC_MAX_PATH_LEN] = "ASCIsol_0000";
    ierr = GetFilename(user.curN,fileLastuASCII);
    ierr = DataSaveASCII(user.x, fileLastuASCII);

    
    PetscInt i,rank;
    FILE *fp;
    char fileTE[PETSC_MAX_PATH_LEN] = "BinaStepAndEnergy";
    MPI_Comm_rank(comm,&rank);
    
    if (!rank) {
        ierr = PetscFOpen(PETSC_COMM_SELF,fileTE,"w",&fp);
        for (i=0;i<user.curN;i++)
        {
            PetscFPrintf(PETSC_COMM_SELF,fp,"%g %g\n",user.stept[i],user.Tenergy[i]);
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
    ierr = VecDestroy(&user.xold2);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "Update"
/*
 * time stepping function
 */
PetscErrorCode Update(void *ptr){
  	AppCtx         	*user = (AppCtx*)ptr;
  	PetscErrorCode 	ierr;
  	ierr = KSPCreate(PETSC_COMM_WORLD,&user->ksp);CHKERRQ(ierr);
	/*----KSP Setting！！！----------------------------------------------*/
	SetKSP(user);		
    /*-------------------------------------------------------------------------------------------*/ 
     /*----B,C,X,b分配空间(x已经分配空间了)----------------------------------------------*/
	ierr = VecDuplicate(user->x,&user->B);CHKERRQ(ierr); 
    ierr = VecDuplicate(user->x,&user->C);CHKERRQ(ierr); 
    ierr = VecDuplicate(user->x,&user->X);CHKERRQ(ierr);
    ierr = VecDuplicate(user->x,&user->Y);CHKERRQ(ierr);
    phi_ex(user);
	CalB(user);
	/*----compute time step------------------------*/
 	Getdt(user);
 	/*-------------------------------------------------------------------------------------------------*/
    CalC(user);
	/*----------------------------------------------------------------------------------------------------------------------------------------*/
    Calnewx(user);
	 /*----------------------------------------------------------------------------------------------------------------------------------------*/ 	
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
	PetscScalar		dt,eps,beta,h;
	PetscReal		pardt,par00,par01,par02,par11;
	DMDALocalInfo	info;
	 
	dt=ts->dt;
	eps=user->eps;
	beta=user->beta;
	
  	ierr = DMDAGetLocalInfo(user->da,&info);CHKERRQ(ierr);
	h=user->Length/info.mx;
	xe = info.xs+info.xm;
  	ye = info.ys+info.ym;
  	switch(user->curN){
  		case 0:
  			pardt=1.0;
  			break;
  		case 1:
  			pardt=2.0/3.0;
  			break;
  		default:
  			pardt=6.0/11.0;
	}
	/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
	//pardt=pardt*2.0;
	/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

    par00=1.0+(4.0*beta/(eps*eps*h*h)+20.0/(h*h*h*h))*pardt*dt*ts->test;
    par01=(-beta/(eps*eps*h*h)-8.0/(h*h*h*h))*pardt*dt*ts->test;
    par11=2.0/(h*h*h*h)*pardt*dt*ts->test;
    par02=1.0/(h*h*h*h)*pardt*dt*ts->test ;
   	/*------------------------------------------*/
	for (j=info.ys; j<ye; j++) {
		for(i=info.xs;i<xe;i++){
			MatStencil	row={0},col[13]={{0}};
			PetscScalar	v[13];
			PetscInt	ncols=0;
			row.j=j;row.i=i;
			/*(0,0)点*/
			col[ncols].j=j;col[ncols].i=i;v[ncols++] = par00;
			/*(0,1)点*/
			col[ncols].j = j;   	col[ncols].i = i-1; v[ncols++] = par01;
			col[ncols].j = j;   	col[ncols].i = i+1; v[ncols++] = par01;
			col[ncols].j = j-1; 	col[ncols].i = i;   v[ncols++] = par01;
			col[ncols].j = j+1; 	col[ncols].i = i;   v[ncols++] = par01;
			/*(0,2)点*/
			col[ncols].j = j;   	col[ncols].i = i-2; v[ncols++] = par02;
			col[ncols].j = j;   	col[ncols].i = i+2; v[ncols++] = par02;
			col[ncols].j = j-2; 	col[ncols].i = i;   v[ncols++] = par02;
			col[ncols].j = j+2; 	col[ncols].i = i;   v[ncols++] = par02;
			/*(1,1)点*/
			col[ncols].j = j-1;   	col[ncols].i = i-1; v[ncols++] = par11;
			col[ncols].j = j+1;   	col[ncols].i = i+1; v[ncols++] = par11;
			col[ncols].j = j-1; 	col[ncols].i = i+1;   v[ncols++] = par11;
			col[ncols].j = j+1; 	col[ncols].i = i-1;   v[ncols++] = par11;
			ierr = MatSetValuesStencil(user->A,1,&row,ncols,col,v,INSERT_VALUES);CHKERRQ(ierr);
		}
	}
	MatAssemblyBegin(user->A,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(user->A,MAT_FINAL_ASSEMBLY);
	/*----KSP的设置，注意其中的矩阵A是定值！！！----------------------------------------------*/
	ierr = KSPSetTolerances(user->ksp,1.e-2/((info.mx+1)*(info.my+1)),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
	ierr = KSPSetFromOptions(user->ksp);CHKERRQ(ierr);
	ierr = KSPSetOperators(user->ksp,user->A,user->A);CHKERRQ(ierr);
	PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "phi_ex"
PetscErrorCode phi_ex(void *ptr){
  	AppCtx         	*user = (AppCtx*)ptr;
  	PetscErrorCode 	ierr;
	PetscInt       	i,j,mx,my,xs,ys,xm,ym,xe,ye;
	PetscReal		phi_old[3];
	Vec				localx,localxold,localxold2;
	
  	PetscFunctionBeginUser;
	
  	ierr = DMDAGetInfo(user->da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  	ierr = DMDAGetCorners(user->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

	xe = xs+xm;
  	ye = ys+ym;
  	
    /*----point to x,B,----------------------------------------------*/
    PetscScalar **phi,**ax_ex,**alocalxold,**alocalxold2,**ax_sum;/*phi=local_x*/
 
    ierr = DMGetLocalVector(user->da,&localx);CHKERRQ(ierr); 
    ierr = VecDuplicate(localx,&localxold);CHKERRQ(ierr);
    ierr = VecDuplicate(localx,&localxold2);CHKERRQ(ierr);
    
    ierr = DMGlobalToLocalBegin(user->da,user->x,INSERT_VALUES,localx);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user->da,user->x,INSERT_VALUES,localx);CHKERRQ(ierr);	
	ierr = DMGlobalToLocalBegin(user->da,user->xold,INSERT_VALUES,localxold);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user->da,user->xold,INSERT_VALUES,localxold);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(user->da,user->xold2,INSERT_VALUES,localxold2);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user->da,user->xold2,INSERT_VALUES,localxold2);CHKERRQ(ierr);
    
    ierr = DMDAVecGetArray(user->da,localx,&phi);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da,user->x_ex,&ax_ex);
    ierr = DMDAVecGetArray(user->da,localxold,&alocalxold);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da,localxold2,&alocalxold2);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da,user->x_sum,&ax_sum);CHKERRQ(ierr);
	
	switch(user->curN){
  		case 0:
  			for (j=ys; j<ye; j++) {
        		for (i=xs; i<xe; i++) {
            		ax_ex[j][i]=phi[j][i];
            		phi_old[0]=phi[j][i];phi_old[1]=phi[j][i];phi_old[2]=phi[j][i];
            		ax_sum[j][i]=BDFi(user,1,phi_old);
        		}
    		}
  			break;
  		case 1:
  			for (j=ys; j<ye; j++) {
        		for (i=xs; i<xe; i++) {
            		ax_ex[j][i]=2*phi[j][i]-alocalxold[j][i];
            		phi_old[0]=phi[j][i];phi_old[1]=alocalxold[j][i];phi_old[2]=alocalxold[j][i];
            		ax_sum[j][i]=BDFi(user,2,phi_old);
        		}
    		}
  			break;
  		default:
  			for (j=ys; j<ye; j++) {
        		for (i=xs; i<xe; i++) {
            		ax_ex[j][i]=3*phi[j][i]-3*alocalxold[j][i]+alocalxold2[j][i];
            		phi_old[0]=phi[j][i];phi_old[1]=alocalxold[j][i];phi_old[2]=alocalxold2[j][i];
            		ax_sum[j][i]=BDFi(user,3,phi_old);
        		}
    		}
	}
 	
 	ierr = DMDAVecRestoreArray(user->da,localx,&phi);CHKERRQ(ierr);
 	ierr = DMDAVecRestoreArray(user->da,user->x_ex,&ax_ex);CHKERRQ(ierr);
 	ierr = DMDAVecRestoreArray(user->da,user->x_sum,&ax_sum);CHKERRQ(ierr);
 	ierr = DMDAVecRestoreArray(user->da,localxold,&alocalxold);CHKERRQ(ierr);
 	ierr = DMDAVecRestoreArray(user->da,localxold2,&alocalxold2);CHKERRQ(ierr);
 	ierr = DMRestoreLocalVector(user->da,&localx);
 	ierr = DMRestoreLocalVector(user->da,&localxold);
 	ierr = DMRestoreLocalVector(user->da,&localxold2);
    
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BDFi"
PetscScalar BDFi(void *ptr,PetscInt n,PetscReal* phi_old){
	AppCtx      *user = (AppCtx*)ptr;
	PetscReal	phi_sum;
	/*phi_old0=phi_n*/
	switch(user->curN){
  		case 0:
  			phi_sum=phi_old[0];
			return phi_sum;
  			break;
  		case 1:
  			phi_sum=4.0/3.0*phi_old[0]-1.0/3.0*phi_old[1];
			return phi_sum;
  			break;
  		default:
  			if(n==3){
				phi_sum=18.0/11.0*phi_old[0]-9.0/11.0*phi_old[1]+2.0/11.0*phi_old[2];
				return phi_sum;
			}
	}
	return 0;	
}

#undef __FUNCT__
#define __FUNCT__ "CalB"
/*
 * 
 */
PetscErrorCode CalB(void *ptr){
  	AppCtx         	*user = (AppCtx*)ptr;
  	PetscErrorCode 	ierr;
	PetscInt       	i,j,mx,my,xs,ys,xm,ym,xe,ye;
	PetscScalar		beta,eps,h;
	PetscReal		rtemp_nplus1_2,rtemp_n,Lapphi,GE1,GE1_nplus1_2,GE,GBxphi;/*temp*/
	Vec				localx;
	//PetcsInt		Ii,Jj;
	
  	ierr = DMDAGetInfo(user->da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  	ierr = DMDAGetCorners(user->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
	
	h=user->Length/mx;
	beta=user->beta;
	eps=user->eps;
	xe = xs+xm;
  	ye = ys+ym;
  	GE=0;
  	GE1=0;
  	GE1_nplus1_2=0;
  	
    /*----point to x,B,----------------------------------------------*/
    PetscScalar **phi,**phi_ex,**aB,**phi_sum;/*phi=local_x*/
 
    ierr = DMGetLocalVector(user->da,&localx);CHKERRQ(ierr); 
    
    ierr = DMGlobalToLocalBegin(user->da,user->x,INSERT_VALUES,localx);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user->da,user->x,INSERT_VALUES,localx);CHKERRQ(ierr);	

    
    ierr = DMDAVecGetArray(user->da,localx,&phi);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da,user->x_ex,&phi_ex);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da,user->x_sum,&phi_sum);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da,user->B,&aB);

    for (j=ys; j<ye; j++) {
        for (i=xs; i<xe; i++) {
            rtemp_nplus1_2=(phi_ex[j][i]*phi_ex[j][i]-1.0-beta);              		
            aB[j][i] = rtemp_nplus1_2*phi_ex[j][i]/(eps*eps);	/*the B is U*/
            GE1_nplus1_2=GE1_nplus1_2+h*h*rtemp_nplus1_2*rtemp_nplus1_2/(eps*eps*4.0);	/*local E1n+1/2*/
            
            rtemp_n=(phi[j][i]*phi[j][i]-1.0-beta); 
            Lapphi=(phi[j+1][i]+phi[j-1][i]+phi[j][i+1]+phi[j][i-1]-4.0*phi[j][i])/(h*h);
            GE1=GE1+h*h*rtemp_n*rtemp_n/(eps*eps*4.0);	/*local E1*/		
			GE=GE+h*h*rtemp_n*rtemp_n/(eps*eps*4.0)
			+h*h*phi[j][i]*(phi[j][i]*beta/(eps*eps)-Lapphi)/2.0;/////////////////这里要用phi_n 
            GBxphi=GBxphi+phi_sum[j][i]*aB[j][i]*h*h;
        }
    }
    ierr = MPI_Allreduce(&GE1_nplus1_2, &user->SE1_nplus1_2, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);	
 	ierr = MPI_Allreduce(&GE1, &user->SE1, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
 	ierr = MPI_Allreduce(&GE, &user->SE, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
 	ierr = MPI_Allreduce(&GBxphi, &user->SBxphi, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
 	
 	
 	switch(user->curN){
  		case 0:
  			user->Rold	=sqrt(user->SE1+user->c0);
 			user->Rold2	=sqrt(user->SE1+user->c0);
 			user->R		=sqrt(user->SE1+user->c0);
  			break;
  		case 1:
  			user->Rold2	=sqrt(user->SE1+user->c0);
  			user->Rold	=user->R;
  			user->R		=sqrt(user->SE1+user->c0);
  			break;
  		default:
  			user->Rold2	=user->Rold;
  			user->Rold	=user->R;
  			user->R		=sqrt(user->SE1+user->c0);
	}
 	user->Tenergy[user->curN] = user->SE;
 	ierr = PetscPrintf(PETSC_COMM_WORLD,"R = %g,E = %g\n",user->R,user->SE);CHKERRQ(ierr);
 	user->SBxphi=user->SBxphi/sqrt(user->SE1_nplus1_2+user->c0);
 			
 	/*----这里才得到真正的B!!!------------------------------*/	
 	ierr = DMDAVecRestoreArray(user->da,localx,&phi);CHKERRQ(ierr);
 	ierr = DMDAVecRestoreArray(user->da,user->B,&aB);CHKERRQ(ierr);
 	ierr = DMDAVecRestoreArray(user->da,user->x_ex,&phi_ex);CHKERRQ(ierr);
 	ierr = DMDAVecRestoreArray(user->da,user->x_sum,&phi_sum);CHKERRQ(ierr);
 	ierr = DMRestoreLocalVector(user->da,&localx);
 	ierr = VecScale(user->B,1./sqrt(user->SE1_nplus1_2+user->c0));CHKERRQ(ierr);
    
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
#define __FUNCT__ "CalC"
/*
 * 
 */
PetscErrorCode CalC(void *ptr){
  	AppCtx         	*user = (AppCtx*)ptr;
  	tsCtx          	*ts = user->ts;
  	PetscErrorCode 	ierr;
	PetscInt       	i,j,mx,my,xs,ys,xm,ym,xe,ye;
	PetscReal		h,R_old[3],R_sum,pardt;
	PetscReal		Lgamma,Ggamma;/*temp*/
	Vec				localB,LapB,localX;
	//PetcsInt		Ii,Jj;
	
  	PetscFunctionBeginUser;
	
  	ierr = DMDAGetInfo(user->da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  	ierr = DMDAGetCorners(user->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
	
	h=user->Length/mx;
	xe = xs+xm;
  	ye = ys+ym;
  	R_old[0]=user->R;R_old[1]=user->Rold;R_old[2]=user->Rold2;
  	switch(user->curN){
  		case 0:
  			pardt=1.0;
  			R_sum=BDFi(user,1,R_old);
  			break;
  		case 1:
  			pardt=2.0/3.0;
  			R_sum=BDFi(user,2,R_old);
  			break;
  		default:
  			pardt=6.0/11.0;
  			R_sum=BDFi(user,3,R_old);
	}
  	
  	
    /*----point to x,B,C----------------------------------------------*/
    PetscScalar **phi_sum,**alocalB,**aC,**aLapB,**alocalX;/*phi=local_x*/ 
    ierr = DMGetGlobalVector(user->da,&LapB);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(user->da,&localB);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(user->da,user->B,INSERT_VALUES,localB);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user->da,user->B,INSERT_VALUES,localB);CHKERRQ(ierr);	
	
    ierr = DMDAVecGetArray(user->da,LapB,&aLapB);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da,localB,&alocalB);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da,user->C,&aC);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da,user->x_sum,&phi_sum);CHKERRQ(ierr);
	 
    
    for (j=ys; j<ye; j++) {
        for (i=xs; i<xe; i++) {           
            aLapB[j][i]=(alocalB[j+1][i]+alocalB[j-1][i]+alocalB[j][i+1]+alocalB[j][i-1]
			-4.0*alocalB[j][i])/(h*h);
            aC[j][i]=phi_sum[j][i]+pardt*ts->dt*ts->test *aLapB[j][i]*(R_sum-1.0/2.0*user->SBxphi);/////////////C还没有改	
        }
    }
    ierr = DMDAVecRestoreArray(user->da,LapB,&aLapB);CHKERRQ(ierr);
    
    ierr = KSPSolve(user->ksp,LapB,user->X);CHKERRQ(ierr);	/*得到的是一个中间参数!!!A-1Gbn,用来算gamma的，暂时用一下X*/
    
	ierr = DMGetLocalVector(user->da,&localX);CHKERRQ(ierr);
	ierr = DMGlobalToLocalBegin(user->da,user->X,INSERT_VALUES,localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user->da,user->X,INSERT_VALUES,localX);CHKERRQ(ierr);	
	ierr = DMDAVecGetArray(user->da,localX,&alocalX);CHKERRQ(ierr);
	for (j=ys; j<ye; j++) {
        for (i=xs; i<xe; i++) {
            Lgamma = alocalB[j][i]*alocalX[j][i];/*这里没加负号*/
            Ggamma = Ggamma + Lgamma*h*h;
        } 
    }	
    		/*----计算 b_n*phi_(n+1)----------------------------*/ 
    ierr = MPI_Allreduce(&Ggamma, &user->gamma, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);
    ierr = DMDAVecRestoreArray(user->da,localX,&alocalX);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(user->da,&localX);
    ierr = VecDestroy(&LapB);CHKERRQ(ierr);
    
 	ierr = DMDAVecRestoreArray(user->da,localB,&alocalB);CHKERRQ(ierr);
 	ierr = DMDAVecRestoreArray(user->da,user->C,&aC);CHKERRQ(ierr);
 	ierr = DMDAVecRestoreArray(user->da,user->x_sum,&phi_sum);CHKERRQ(ierr);
 	ierr = DMRestoreLocalVector(user->da,&localB);
 	
	
    ierr = KSPSolve(user->ksp,user->C,user->Y);CHKERRQ(ierr);	/*得到的是一个中间参数X*/
		   
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
	PetscInt       	i,j,mx,my,xs,ys,xm,ym,xe,ye;
	PetscReal		h,pardt;
	PetscReal		LBxphiP,GBxphiP;/*temp*/
	Vec				localB,localY;
	//PetcsInt		Ii,Jj;
	
  	PetscFunctionBeginUser;
	
  	ierr = DMDAGetInfo(user->da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  	ierr = DMDAGetCorners(user->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
	
	h=user->Length/mx;
	xe = xs+xm;
  	ye = ys+ym;
  	switch(user->curN){
  		case 0:
  			pardt=1.0;
  			break;
  		case 1:
  			pardt=2.0/3.0;
  			break;
  		default:
  			pardt=6.0/11.0;
	}
  	
    /*----point to x,B,C----------------------------------------------*/
    PetscScalar **alocalB,**alocalY;/*phi=local_x*/ 
     
    ierr = DMGetLocalVector(user->da,&localY);CHKERRQ(ierr);
	ierr = VecDuplicate(localY,&localB);CHKERRQ(ierr);
	
    ierr = DMGlobalToLocalBegin(user->da,user->B,INSERT_VALUES,localB);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user->da,user->B,INSERT_VALUES,localB);CHKERRQ(ierr);	
    ierr = DMGlobalToLocalBegin(user->da,user->Y,INSERT_VALUES,localY);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user->da,user->Y,INSERT_VALUES,localY);CHKERRQ(ierr);	
	
    ierr = DMDAVecGetArray(user->da,localB,&alocalB);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da,localY,&alocalY);CHKERRQ(ierr);
	 
	GBxphiP = 0.0;	/*初始化 b_n*phi_(n+1) */ 
    for (j=ys; j<ye; j++) {
        for (i=xs; i<xe; i++) {
            LBxphiP = alocalB[j][i]*alocalY[j][i]/(1-ts->dt*pardt*user->gamma*ts->test/2.0);
            GBxphiP = GBxphiP + LBxphiP*h*h;
        } 
    }	
    		/*----计算 b_n*phi_(n+1)----------------------------*/ 
    ierr = MPI_Allreduce(&GBxphiP, &user->SBxphiP, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);
    		
    		/*----计算最后的方程------------------*/ 
	VecAXPY(user->Y,ts->dt*ts->test*pardt*user->SBxphiP/2.0,user->X);

    ierr = DMDAVecRestoreArray(user->da,localB,&alocalB);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da,localY,&alocalY);CHKERRQ(ierr);
 	ierr = DMRestoreLocalVector(user->da,&localB);
 	ierr = DMRestoreLocalVector(user->da,&localY);
 	
 	ierr = VecCopy(user->xold,user->xold2);
 	ierr = VecCopy(user->x,user->xold);
    ierr = VecCopy(user->Y,user->x);		
    	 /*----------------------------------------------------------------------------------------------------------------------------------------*/
      /*----------------------------------------------------------------------------------------------------------------------------------------*/
	PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitial"
/*
 FormInitialGuess - Forms initial values.
 
 Output Parameter:
 user.Xpre - vector
 */
PetscErrorCode FormInitial(DM da,AppCtx *user)
{
    PetscInt       i,j,Mx,My,xs,ys,xm,ym,use_random=0;
    PetscErrorCode ierr;
    PetscReal      L,hx,hy;
    //phi=sqrt(-user->rconstant)/2.0,Aphi=0.333050218204464,qx=sqrt(3)/2.0,qy=sqrt(3)/2.0;
    PetscScalar    **x;
    PetscBool      flg = PETSC_FALSE;
    PetscFunctionBeginUser;
    if (flg)
    {
        char  filename[PETSC_MAX_PATH_LEN] = "Initial/Initial_rand_L128_N256";
        ierr = DataLoadBin(user->X, filename);
    }
    else
    {
        ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
		if (use_random)
		{
	    	PetscRandom rctx;
	    	PetscReal sum,mean;
        	PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
	    	PetscRandomSetFromOptions(rctx);
	    	PetscRandomSetInterval(rctx,-0.15,0.65);
	    	VecSetRandom(user->x,rctx);
            VecSum(user->x,&sum);
	    	mean=sum/(double)(Mx*My);
	    	VecScale(user->x,0.25/mean);
		}
		else
		{
	    	PetscReal lx,ly;
	    	L = user->Length;
        	hx = L/(PetscReal)Mx;
        	hy = L/(PetscReal)My;
        
	    	VecSet(user->x,0);
        	ierr = DMDAVecGetArray(da,user->x,&x);CHKERRQ(ierr);
        	ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
            
        	for (j=ys; j<ys+ym; j++) {
				for (i=xs; i<xs+xm; i++) {
					lx=i*hx;
					ly=j*hy;
		        	x[j][i]=0.05*sin(lx)*sin(ly);
				}
	    	}
            ierr = DMDAVecRestoreArray(da,user->x,&x);CHKERRQ(ierr);
		}    
    }
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DataSaveVtk"
PetscErrorCode DataSaveVtk(Vec x, char *filename)
{
    PetscErrorCode ierr;
    PetscViewer    dataviewer;
    
    PetscFunctionBegin;
    
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&dataviewer);CHKERRQ(ierr);
    ierr = VecView(x,dataviewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&dataviewer); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DataLoadVtk"
PetscErrorCode DataLoadVtk(Vec x, char *filename)
{
    PetscErrorCode ierr;
    PetscViewer    dataviewer;
    
    PetscFunctionBegin;
    
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&dataviewer);CHKERRQ(ierr);
    ierr = VecLoad(x,dataviewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&dataviewer); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DataSaveBin"
PetscErrorCode DataSaveBin(Vec x, char *filename)
{
    PetscErrorCode ierr;
    PetscViewer    dataviewer;
    
    PetscFunctionBegin;
    
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&dataviewer); CHKERRQ(ierr);
    ierr = VecView(x,dataviewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&dataviewer); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DataLoadBin"
PetscErrorCode DataLoadBin(Vec x, char *filename)
{
    PetscErrorCode ierr;
    PetscViewer    dataviewer;
    
    PetscFunctionBegin;
    
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&dataviewer); CHKERRQ(ierr);
    ierr = VecLoad(x,dataviewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&dataviewer); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DataSaveASCII"

PetscErrorCode DataSaveASCII(Vec x, char *filename)
{
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
PetscErrorCode DataLoadASCII(Vec x, char *filename)
{
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
PetscErrorCode GetFilename(PetscInt i, char *filename)
{
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
PetscScalar max(PetscScalar a, PetscScalar b)
{
    if (a>b){return a;}
    else{return b;}
}

#undef __FUNCT__
#define __FUNCT__ "min"
PetscScalar min(PetscScalar a, PetscScalar b)
{
    if (a>b){return b;}
    else{return a;}
}
