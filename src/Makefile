#
# Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

include ../makefiles/common.mk
include ../makefiles/version.mk

##### src files
INCEXPORTS  := nccl.h nccl_net.h
LIBSRCFILES := init.cc channel.cc bootstrap.cc transport.cc enqueue.cc enqueue_ofccl.cc group.cc debug.cc proxy.cc enhcompat.cc net.cc \
		misc/nvmlwrap.cc misc/ibvwrap.cc misc/gdrwrap.cc misc/utils.cc misc/argcheck.cc misc/socket.cc misc/shmutils.cc misc/profiler.cc misc/param.cc \
		transport/p2p.cc transport/shm.cc transport/net.cc transport/net_socket.cc transport/net_ib.cc transport/coll_net.cc \
                collectives/sendrecv.cc collectives/all_reduce.cc collectives/all_gather.cc collectives/broadcast.cc collectives/reduce.cc collectives/reduce_scatter.cc \
								collectives_ofccl/ofccl_all_reduce.cc \
                graph/topo.cc graph/paths.cc graph/search.cc graph/connect.cc graph/rings.cc graph/trees.cc graph/tuning.cc graph/xml.cc

ENQ_DEVSRC 	:= enqueue_ofccl_dev.cu
##### lib files
LIBNAME     := libnccl.so
STATICLIBNAME := libnccl_static.a
##### pkgconfig files
PKGCONFIGFILE := nccl.pc
##### dirs
BUILDDIR ?= $(abspath ../build)
INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj
PKGDIR := $(BUILDDIR)/lib/pkgconfig
##### target files
CUDARTLIB  ?= cudart_static
INCTARGETS := $(INCEXPORTS:%=$(INCDIR)/%) # 替换出来include的目标，即incdir下的相应文件。
LIBSONAME  := $(LIBNAME:%=%.$(NCCL_MAJOR))
LIBTARGET  := $(LIBNAME:%=%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH)) # 替换libname，加了后缀
STATICLIBTARGET := $(STATICLIBNAME)
PKGTARGET  := $(PKGCONFIGFILE)
LIBOBJ     := $(LIBSRCFILES:%.cc=$(OBJDIR)/%.o) # 替换出来libobj的target，objdir下的.o文件。这里我们需要加一个cu文件编译出来的obj，即enqueue_ofccl.cu的编译结果。
DEPFILES   := $(LIBOBJ:%.o=%.d) # 为了解决include依赖，对所有.o文件生成.d文件来记录相应的依赖
LDFLAGS    += -L${CUDA_LIB} -l$(CUDARTLIB) -lpthread -lrt -ldl

DEVICELIB  := $(BUILDDIR)/obj/collectives/device/colldevice.a $(BUILDDIR)/obj/collectives_ofccl/device/colldevice_ofccl.a # device的相关函数，做一个静态库就够了。

ENQ_DEVOBJ := $(ENQ_DEVSRC:%.cu=$(OBJDIR)/%.o)
ENQ_DEVDEP := $(ENQ_DEVSRC:%.cu=$(OBJDIR)/%.dep)

ENQ_NVCUFLAGS = $(NVCUFLAGS)
ENQ_NVCUFLAGS += -I. -I.. -Iinclude -I$(BUILDDIR)/include --compiler-options "-fPIC -fvisibility=hidden"
##### rules
build : lib staticlib

lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET) $(PKGDIR)/$(PKGTARGET)

staticlib : $(LIBDIR)/$(STATICLIBTARGET)

$(DEVICELIB): ALWAYS_REBUILD $(INCTARGETS)
	$(MAKE) -C collectives/device
	$(MAKE) -C collectives_ofccl/device

# Empty target to force rebuild
ALWAYS_REBUILD:

-include $(DEPFILES)
$(LIBDIR)/$(LIBTARGET) $(LIBDIR)/$(STATICLIBTARGET) : $(LIBOBJ)

$(INCDIR)/nccl.h : nccl.h.in
# NCCL_VERSION(X,Y,Z) ((X) * 10000 + (Y) * 100 + (Z))
	@$(eval NCCL_VERSION := $(shell printf "%d%02d%02d" $(NCCL_MAJOR) $(NCCL_MINOR) $(NCCL_PATCH)))
	mkdir -p $(INCDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${nccl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${nccl:Version}/$(NCCL_VERSION)/g" \
	    $< > $@

$(LIBDIR)/$(LIBTARGET): $(LIBOBJ) $(DEVICELIB) $(ENQ_DEVOBJ)
	@printf "Linking    %-35s > %s\n" $(LIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LIBOBJ) $(DEVICELIB) $(ENQ_DEVOBJ) $(LDFLAGS)
# ln [OPTION]... [-T] TARGET LINK_NAME, 下边这两行不用动。
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)
	@printf "Link Success    %-35s > %s\n" $(LIBTARGET) $@

null :=
space := $(null) #
comma := ,

$(LIBDIR)/$(STATICLIBTARGET): $(LIBOBJ) $(DEVICELIB) $(ENQ_DEVOBJ)
	@printf "HELLO Archiving  %-35s > %s\n" $(STATICLIBTARGET) $@
	mkdir -p $(LIBDIR)
	printf "create $@\naddlib $(DEVICELIB)\naddmod $(subst $(space),$(comma),$(strip $(LIBOBJ)))\nsave\nend" | ar -M

$(ENQ_DEVOBJ) : $(ENQ_DEVSRC) $(ENQ_DEVDEP)
	@printf "ENQUEUE DEV Compiling  %-35s > %s\\n" $< $@
	mkdir -p $(OBJDIR)
	${NVCC} ${ENQ_NVCUFLAGS} -dc $< -o $@

$(ENQ_DEVDEP) : $(ENQ_DEVSRC)
	mkdir -p $(OBJDIR)
	@$(NVCC) $(ENQ_NVCUFLAGS) -M $< -o $@.tmp
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $@.tmp > $@
	@sed -e 's/.*://' -e 's/\\$$//' < $@.tmp | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $@
	@rm -f $@.tmp
	@cp $@ $(@:.dep=.d)

$(PKGDIR)/nccl.pc : nccl.pc.in
	mkdir -p $(PKGDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e 's|$${nccl:Prefix}|\$(PREFIX)|g' \
	    -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    $< > $@

$(INCDIR)/%.h : %.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(INCDIR)/nccl_%.h : include/nccl_%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(PKGDIR)/%.pc : %.pc
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(PKGDIR)
	install -m 644 $< $@

$(OBJDIR)/%.o : %.cc $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) -Iinclude -c $< -o $@
	@$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) -Iinclude -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

clean :
	$(MAKE) -C collectives/device clean
	$(MAKE) -C collectives_ofccl/device clean
	rm -rf ${INCDIR} ${LIBDIR} ${PKGDIR} ${OBJDIR}

install : build
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/lib/pkgconfig
	mkdir -p $(PREFIX)/include
	cp -P -v $(BUILDDIR)/lib/lib* $(PREFIX)/lib/
	cp -P -v $(BUILDDIR)/lib/pkgconfig/* $(PREFIX)/lib/pkgconfig/
	cp -v $(BUILDDIR)/include/* $(PREFIX)/include/

FILESTOFORMAT := $(shell find . -name ".\#*" -prune -o \( -name "*.cc" -o -name "*.h" \) -print | grep -v -E 'ibvwrap.h|nvmlwrap.h|gdrwrap.h|nccl.h')
# Note that formatting.mk defines a new target so in order to not overwrite the default target,
# it shouldn't be included at the top. Also, it uses the above definition of FILESTOFORMAT as well
# as the BUILDDIR variable.
include ../makefiles/formatting.mk
