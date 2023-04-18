FC=f95
LIB=gslib/gslib.a

PROGS = \
	addcoord \
	anneal \
	backtr \
	bicalib \
	bigaus \
	bivplt \
	cokb3d \
	declus \
	draw \
	ellipsim \
	gam \
	gamv \
	gtsim \
	histplt \
	histsmth \
	ik3d \
	kb2d \
	kt3d \
	locmap \
	lusim \
	nscore \
	pfsim \
	pixelplt \
	plotem \
	postik \
	postsim \
	probplt \
	qpplt \
	rotcoord \
	sasim \
	scatplt \
	scatsmth \
	sgsim \
	sisim \
	sisim_gs \
	sisim_lm \
	trans \
	vargplt \
	varmap \
	vmodel

all: $(LIB) $(PROGS)

$(LIB):
	cd gslib; make; cd ..

%: %.for
	$(FC) -o $@ $< $(LIB)

clean:
	cd gslib; make clean; cd ..; rm $(PROGS)
