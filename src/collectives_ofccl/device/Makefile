
include ../../../makefiles/common.mk
include ../../../makefiles/version.mk

BUILDDIR ?= $(abspath ../../../build)
OBJDIR := $(BUILDDIR)/obj/collectives_ofccl/device
# ↑ 改了名字

LIBSRCFILES := ofccl_all_reduce.cu ofccl_broadcast.cu ofccl_reduce.cu ofccl_all_gather.cu ofccl_reduce_scatter.cu ofccl_sendrecv.cu ofccl_onerank_reduce.cu

LIBSRCFILES += functions_ofccl.cu

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

LIBOBJ     := $(GENOBJS) $(OBJDIR)/functions_ofccl.o $(OBJDIR)/onerank_reduce.o

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

# Compiled kernels and collectives with relocatable device code ... 
$(OBJDIR)/functions_ofccl.o : functions_ofccl.cu $(OBJDIR)/functions_ofccl.dep
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(NVCUFLAGS) -dc $< -o $@

$(OBJDIR)/onerank_reduce.o : onerank_reduce.cu $(OBJDIR)/onerank_reduce.dep
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(NVCUFLAGS) -dc $< -o $@

# ... and create the device-side linked object with all those.
$(DEVOBJ) : $(LIBOBJ)
	$(NVCC) $(NVCUFLAGS) -dlink $^ -o $@

clean:
	rm -f $(LIBOBJ) $(DEVOBJ) $(DEPFILES) $(DEPENDFILES) $(RULESFILE) $(STATICLIB)