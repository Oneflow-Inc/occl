
include ../../../makefiles/common.mk
include ../../../makefiles/version.mk

BUILDDIR ?= $(abspath ../../../build)
OBJDIR := $(BUILDDIR)/obj/collectives_ofccl/device
# ↑ 改了名字

LIBSRCFILES := ofccl_all_reduce.cu
# ↑ 原来的版本：
# LIBSRCFILES := all_reduce.cu broadcast.cu reduce.cu all_gather.cu reduce_scatter.cu sendrecv.cu onerank_reduce.cu

# LIBSRCFILES += functions.cu

DEPFILES   := $(patsubst %.cu, $(OBJDIR)/%.d, $(LIBSRCFILES))
DEPENDFILES:= $(DEPFILES:%.d=%.dep)
STATICLIB  := $(OBJDIR)/colldevice_ofccl.a
# ↑ 改了名字
DEVOBJ     := $(OBJDIR)/devlink_ofccl.o
# ↑ 改了名字
RULESFILE  := $(OBJDIR)/Makefile_ofccl.rules
# ↑ 改了名字

NVCUFLAGS  += -I. -I.. -I$(BUILDDIR)/include -I../../include --compiler-options "-fPIC -fvisibility=hidden"


all: $(STATICLIB)

# Dummy rule so that the extra dependency (%.dep) files are preserved by make
all_deps: $(DEPENDFILES)

# Auto-generating the rules per op/reduction/datatype/algorithm
$(RULESFILE) :
	@printf "Generating %-35s > %s\n" rules $@
	@mkdir -p $(OBJDIR)
	@CUDA_MAJOR=${CUDA_MAJOR} CUDA_MINOR=${CUDA_MINOR} ./gen_rules.sh $(OBJDIR) > $@

-include $(RULESFILE)

LIBOBJ     := $(GENOBJS)
# ↑ 原来的版本：
# LIBOBJ     := $(GENOBJS) $(OBJDIR)/functions.o $(OBJDIR)/onerank_reduce.o

-include $(DEPFILES)

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

# Compiled kernels and collectives with relocatable device code ... 目前下边这俩都没有
# $(OBJDIR)/functions.o : functions.cu $(OBJDIR)/functions.dep
# 	@printf "Compiling  %-35s > %s\n" $< $@
# 	mkdir -p `dirname $@`
# 	$(NVCC) $(NVCUFLAGS) -dc $< -o $@

# $(OBJDIR)/onerank_reduce.o : onerank_reduce.cu $(OBJDIR)/onerank_reduce.dep
# 	@printf "Compiling  %-35s > %s\n" $< $@
# 	mkdir -p `dirname $@`
# 	$(NVCC) $(NVCUFLAGS) -dc $< -o $@

# ... and create the device-side linked object with all those.
$(DEVOBJ) : $(LIBOBJ)
	$(NVCC) $(NVCUFLAGS) -dlink $^ -o $@

clean:
	rm -f $(LIBOBJ) $(DEVOBJ) $(DEPFILES) $(DEPENDFILES) $(RULESFILE) $(STATICLIB)