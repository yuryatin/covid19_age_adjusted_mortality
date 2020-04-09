deathcurve.so: deathcurve.c
	clang -fPIC -shared -pthread -Wall -o libdeathcurve.so deathcurve.c -lm -lpthread
