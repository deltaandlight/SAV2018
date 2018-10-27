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

extern PetscErrorCode FormInitial(DM, AppCtx*);
extern PetscErrorCode Update(void *ptr);
extern PetscErrorCode SetKSP(void *ptr);
extern PetscErrorCode CalB(void *ptr);
extern PetscErrorCode CalC(void *ptr);
extern PetscErrorCode Calnewx(void *ptr);
extern PetscErrorCode Getdt(void *ptr);
extern PetscErrorCode phi_ex(void *ptr);

extern PetscErrorCode DataSaveASCII(Vec, char*);
extern PetscErrorCode DataLoadASCII(Vec, char*);
extern PetscErrorCode GetFilename(PetscInt, char*);
extern PetscScalar    max(PetscScalar, PetscScalar);
extern PetscScalar    min(PetscScalar, PetscScalar);

extern PetscErrorCode OutputInfoAndConfig(int, char**);
