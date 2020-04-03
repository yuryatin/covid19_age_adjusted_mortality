deathcurve.so: deathcurve.c
	clang -shared -pthread -Wall -o deathcurve.so deathcurve.c
