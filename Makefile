CC=g++
CFLAGS= 
LFLAGS= -Wall -larmadillo 
OBJS = main.o Davidson.o 

arma_davidson: $(OBJS) main.cpp Davidson.cpp Davidson.h
	$(CC) $(OBJS) $(LIBS) -o arma_davidson $(LFLAGS) 

clean:
	\rm *.o arma_davidson 

