
CFLAGS           =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =

export PETSC_DIR=/home/yangchao/zl/petsc-3.6.4
export PETSC_ARCH=linux-gnu-c-debug
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

objects = main.o initial.o update.o set_ksp.o phi_ex.o\
	  cal.o get_dt.o output.o extra_func.o

main: $(objects) chkopts
	-${CLINKER} -o main $(objects) ${PETSC_LIB}
	${RM} $(objects)

$(objects):def.h

.PHONY: sol_clean all_clean

sol_clean: 
	-${RM} *sol*
all_clean:
	-${RM} *sol* slurm*
