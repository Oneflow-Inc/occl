
include ../../../makefiles/common.mk
include ../../../makefiles/version.mk

BUILDDIR ?= $(abspath ../../../build)
OBJDIR := $(BUILDDIR)/obj/collectives_ofccl/device
# ↑ 改了名字

LIBSRCFILES := ofccl_all_reduce.cu ofccl_broadcast.cu ofccl_reduce.cu ofccl_all_gather.cu ofccl_reduce_scatter.cu ofccl_sendrecv.cu ofccl_onerank_reduce.cu

LIBSRCFILES += functions_ofccl.cu

ENQ_DEVSRC 	:= enqueue_ofccl_dev.cu

DEPFILES   := $(patsubst %.cu, $(OBJDIR)/%.d, $(LIBSRCFILES))
DEPENDFILES:= $(DEPFILES:%.d=%.dep)
STATICLIB  := $(OBJDIR)/colldevice_ofccl.a
# ↑ 改了名字
DEVOBJ     := $(OBJDIR)/devlink_ofccl.o
# ↑ 改了名字
RULESFILE  := $(OBJDIR)/Makefile_ofccl.rules
# ↑ 改了名字

NVCUFLAGS  += -I. -I.. -I$(BUILDDIR)/include -I../../include --compiler-options "-fPIC -fvisibility=hidden"

ENQ_DEVOBJ 			:= $(ENQ_DEVSRC:%.cu=$(OBJDIR)/%.o)
ENQ_DEVDEP 			:= $(ENQ_DEVSRC:%.cu=$(OBJDIR)/%.dep)
# ENQ_DEV_LINKOBJ := $(ENQ_DEVSRC:%.cu=$(OBJDIR)/%_link.o)
# ENQ_DEVSTATIC			:= $(ENQ_DEVSRC:%.cu=$(OBJDIR)/%.a)

DEBUG_ENQ ?= 0
ENQ_NVCUFLAGS := -ccbin $(CXX) $(NVCC_GENCODE) -std=c++11 --expt-extended-lambda -Xptxas -maxrregcount=96 -Xfatbin -compress-all
ENQ_NVCUFLAGS += ${GCOV_FLAGS:%=-Xcompiler %}
ifeq ($(DEBUG_ENQ), 0)
ENQ_NVCUFLAGS += -O3
else
ENQ_NVCUFLAGS += -O0 -G -g
endif
ifneq ($(VERBOSE), 0)
ENQ_NVCUFLAGS += -Xptxas -v -Xcompiler -Wall,-Wextra,-Wno-unused-parameter
else
.SILENT:
endif
ifneq ($(KEEP), 0)
ENQ_NVCUFLAGS += -keep
endif
ENQ_NVCUFLAGS += -I. -I.. -I$(BUILDDIR)/include -I../../include --compiler-options "-fPIC -fvisibility=hidden"


all: $(STATICLIB)

# Dummy rule so that the extra dependency (%.dep) files are preserved by make
all_deps: $(DEPENDFILES)

# Auto-generating the rules per op/reduction/datatype/algorithm
$(RULESFILE) :
	@printf "Generating %-35s > %s\n" rules $@
	@mkdir -p $(OBJDIR)
	@CUDA_MAJOR=${CUDA_MAJOR} CUDA_MINOR=${CUDA_MINOR} ./gen_rules.sh $(OBJDIR) > $@

-include $(RULESFILE)

LIBOBJ     := $(GENOBJS) $(OBJDIR)/functions_ofccl.o $(OBJDIR)/ofccl_onerank_reduce.o $(ENQ_DEVOBJ)

-include $(DEPFILES)

# 应该把enq_dev.cu编译出来的.o放到LIBOBJ里边。人家原来的LIBOBJ DEVOBJ依赖的源文件是一样的，咱们假如enq_dev.cu之后应该保持这个性质。事实上只把enq_dev.cu编译出来的.o放到DEVOBJ里边，遭遇了运行时的libnccl.so.2: undefined symbol: _Z12daemonKernelP2SQP2CQiiP3CQEPiS5_S5_P16DevComm7WorkElemP14ofcclShmemData报错
$(STATICLIB): $(LIBOBJ) $(DEVOBJ)
	@printf "Archiving  %-35s > %s\n" objects $@
	ar cr $@ $^

# We do not want make to build *.d when running make clean.
# So we only provide targets for .dep which will produce .dep and .d,
# with only .d being included, and .dep keeping track of what needs to
# be regenerated.
$(OBJDIR)/%.dep : %.cu
	@mkdir -p $(OBJDIR)
	@$(NVCC) $(NVCUFLAGS) -M $< -o $@.tmp
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $@.tmp > $@
	@sed -e 's/.*://' -e 's/\\$$//' < $@.tmp | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $@
	@rm -f $@.tmp
	@cp $@ $(@:.dep=.d)

# Compiled kernels and collectives with relocatable device code ... 
$(OBJDIR)/functions_ofccl.o : functions_ofccl.cu $(OBJDIR)/functions_ofccl.dep
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(NVCUFLAGS) -dc $< -o $@

$(OBJDIR)/ofccl_onerank_reduce.o : ofccl_onerank_reduce.cu $(OBJDIR)/ofccl_onerank_reduce.dep
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(NVCUFLAGS) -dc $< -o $@

# ... and create the device-side linked object with all those.
$(DEVOBJ) : $(LIBOBJ)
	$(NVCC) $(NVCUFLAGS) -dlink $^ -o $@

$(ENQ_DEVDEP) : $(ENQ_DEVSRC)
	mkdir -p $(OBJDIR)
	@$(NVCC) $(ENQ_NVCUFLAGS) -M $< -o $@.tmp
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $@.tmp > $@
	@sed -e 's/.*://' -e 's/\\$$//' < $@.tmp | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $@
	@rm -f $@.tmp
	@cp $@ $(@:.dep=.d)

-include $(ENQ_DEVDEP)
$(ENQ_DEVOBJ) : $(ENQ_DEVSRC) $(ENQ_DEVDEP)
	@printf "ENQUEUE DEV Compiling  %-35s > %s\\n" $< $@
	@printf "DEBUG_ENQ is %s\\n" $(DEBUG_ENQ)
	@printf "ENQ_NVCUFLAGS is %s\\n" $(ENQ_NVCUFLAGS)
	mkdir -p $(OBJDIR)
	${NVCC} ${ENQ_NVCUFLAGS} -dc $< -o $@

clean:
	rm -f $(LIBOBJ) $(DEVOBJ) $(DEPFILES) $(DEPENDFILES) $(RULESFILE) $(STATICLIB)