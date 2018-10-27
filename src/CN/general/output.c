#include <petscksp.h>
#include <petscdmda.h>
#include <petscdm.h>
#include <petsctime.h>
#include <math.h>
#include <time.h>
#include "def.h"

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
