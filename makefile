BINS = diffusion 
CFLAGS = -O2 -lOpenCL

.PHONY : all
all : $(BINS) 

diffusion : diffusion.c
	gcc $(CFLAGS) -o $@ $<

.PHONY : clean
clean :
	rm -rf $(BINS)
