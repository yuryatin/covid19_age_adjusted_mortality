/*
This code and the code of the Python wrapper module may help you
fit an analytically expressed function to the COVID-19 mortality
data to model age-adjusted mortality risk using maximum likelihood
point estimates.

This software was desinged in two parts: a Python wrapper script to
help more easily import and transform the input tabular data and
finally plot the fitted curves, and a shared C library to dramatically
speed up the calculation.

The shared C library provides the opportunity to test arbitraty
functions on condition that, in the domain between 0 and 120+, they
return natural logarithms of values between 0.0 and 1.0 - otherwise
in this scenario, it will make no sense.

The C library interface fitFunction() function accepts three arrays
and two other parameters:
- array of subjects' ages (of the data type 'double' to accomodate
    data that specify full dates of birth instead of years of birth)
- array of subjects' outcomes, where '1' is death and '0' is a more
positive outcome (of the data type 'int')
- the length of those arrays (of the data type 'int')
- array of 16 double-precision floats to accomodate the calculated
    parameters the interface function returns
- a pointer to an 'int' variable to return the number of the second
    best fit function, if necessary
So the Python script can be modified to supply case-by-case data I
don't yet have access to.

Copyright (C) 2020  Alexander Yuryatin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <pthread.h>
#include <stdint.h>

/*The macro below was created along the similar function internalLogF() to avoid function call overhead when this code's execution is required for one time only*/
#define internalLogM(x, b0, b1, b2, b3, b4, b5, b6, b7) (log(b0 + b1 * x + b2 * pow(x, 2.0) + b3 * pow(x, 3.0) + b4 * pow(x, 4.0) + b5 * pow(x, 5.0) + b6 * pow(x, 6.0) + b7 * pow(x, 7.0)))
#define logVerified(x) ( x <= 0.0 ? -DBL_MAX : log(x) )
#define indexConverter(x) (1 + (x > 0 ? (int)pow(3,1) : 0) + (x > 1 ? (int)pow(3,2) : 0) + (x > 2 ? (int)pow(3,3) : 0) + (x > 3 ? (int)pow(3,4) : 0) + (x > 4 ? (int)pow(3,5) : 0) + (x > 5 ? (int)pow(3,6) : 0) + (x > 6 ? (int)pow(3,7) : 0))
#define THREADS_MAX 6561   // 3 ^ 8 - the former is the number of tests for each parameter per step, the latter is the number of fitted parameters. So many threads don't significantly impede performance in practice (though you may want to reassess that) but will use whatever number of CPU cores and threads your laptop, workstation, or server has.

/* these variables were made global to make them accessible (read-only) to all threads */
int func_g, * outcome_g, length_g, precision_g;
double b0_g, b1_g, b2_g, b3_g, b4_g, b5_g, b6_g, b7_g, * age_g, * result_g;

/*The function below was created along the similar macro internalLogM() to sacrifice function call overhead when this code is required to execute for more than once with the same arguments */
double internalLogF(double x, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7) {
    return log(b0 + b1 * x + b2 * pow(x, 2.0) + b3 * pow(x, 3.0) + b4 * pow(x, 4.0) + b5 * pow(x, 5.0) + b6 * pow(x, 6.0) + b7 * pow(x, 7.0));
}

/* The five fitted functions */
double erfLog(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7){
    double result = erf(internalLogM(x, b0, b1, b2, b3, b4, b5, b6, b7)) * 0.5 + 0.5;
    return outcome ? logVerified(result) : logVerified(1.0 - result);
}
    
double hyperbTan(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7){
    double result = tanh(internalLogM(x, b0, b1, b2, b3, b4, b5, b6, b7)) * 0.5 + 0.5;
    return outcome ? logVerified(result) : logVerified(1.0 - result);
}

double GudFunc(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7){
    double result = atan(tanh(internalLogM(x, b0, b1, b2, b3, b4, b5, b6, b7))) * M_1_PI * 2.0 + 0.5;
    return outcome ? logVerified(result) : logVerified(1.0 - result);
}

double xOverX2(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7){
    double temp = internalLogF(x, b0, b1, b2, b3, b4, b5, b6, b7);
    double result = temp * pow(1.0 + pow(temp, 2.0), -0.5) * 0.5 + 0.5;
    return outcome ? logVerified(result) : logVerified(1.0 - result);
}

double xOverAbs(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7){
    double temp = internalLogF(x, b0, b1, b2, b3, b4, b5, b6, b7);
    double result = temp / (1 +fabs(temp)) * 0.5 + 0.5;
    return outcome ? logVerified(result) : logVerified(1.0 - result);
}

/* array of function pointers to facilitate their calls by numbers */
double (*testFunc[5])(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7);

void * getML(void * threadId) {
    result_g[(int)threadId] = 0.0;
    int b0_l = (int)threadId / 2187 - 1;
    int b1_l = (int)threadId % 2187 / 729 - 1;
    int b2_l = (int)threadId % 729 / 243 - 1;
    int b3_l = (int)threadId % 243 / 81 - 1;
    int b4_l = (int)threadId % 81 / 27 - 1;
    int b5_l = (int)threadId % 27 / 9 - 1;
    int b6_l = (int)threadId % 9 / 3 - 1;
    int b7_l = (int)threadId % 3 - 1;
    double precision_l = pow(10.0, -precision_g);
    for (int i = 0; i < length_g; ++i)
        result_g[(int)threadId] += testFunc[func_g](age_g[i], outcome_g[i], pow(10.0, b0_g + b0_l * precision_l),
                                            pow(10.0, b1_g + b1_l * precision_l),
                                            pow(10.0, b2_g + b2_l * precision_l),
                                            pow(10.0, b3_g + b3_l * precision_l),
                                            pow(10.0, b4_g + b4_l * precision_l),
                                            pow(10.0, b5_g + b5_l * precision_l),
                                            pow(10.0, b6_g + b6_l * precision_l),
                                            pow(10.0, b7_g + b7_l * precision_l));
    return (void *) (intptr_t) 0;
}

int oneStep(int func, double * result, double * age, int * outcome, int length, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7, int precision) {
    func_g = func;    outcome_g = outcome;    length_g = length;    precision_g = precision;
    b0_g = b0;    b1_g = b1;    b2_g = b2;    b3_g = b3;    b4_g = b4;    b5_g = b5;    b6_g = b6;    b7_g = b7;    age_g = age;
    /* "_g" on the end of the variable's name attempts to remind the coder that the variable is global */
    pthread_attr_t attr;
    size_t stacksize = 16384;    // limited due to a huge number of generated threads to potentially preserve RAM
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, stacksize);
    pthread_t * myThreads;
    result_g = malloc(pow(3,8) * sizeof(double));
    myThreads = malloc(pow(3,8) * sizeof(pthread_t));
    int threadCount=0;
    for (threadCount=0; threadCount < THREADS_MAX; ++threadCount)
        pthread_create(&myThreads[threadCount], NULL, getML, (void *) (intptr_t) threadCount);
    for (int i = 0; i < threadCount; ++i)
        pthread_join(myThreads[i], NULL);
    * result = result_g[1];
    int position = 1;
    int condition = 1;
    while(condition) {
        condition = 0;
        for (int i = 1; i < 8; ++i) {
            if (result_g[indexConverter(i)] >= *result) {
                * result = result_g[indexConverter(i)];
                position = indexConverter(i);
            }
        }
        for (int i = 0; i < THREADS_MAX; ++i) {
            if (result_g[i] > *result) {
                * result = result_g[i];
                position = i;
                condition = 1;
            }
        }
    }
    free(myThreads);
    free(result_g);
    return position;
}

/* the function that needs to be called from Python script */
int fitFunction(double * ages, int * the_outcomes, int length, double * output, int * secondRes) {
    testFunc[0] = &erfLog;
    testFunc[1] = &hyperbTan;
    testFunc[2] = &GudFunc;
    testFunc[3] = &xOverX2;
    testFunc[4] = &xOverAbs;
    double finalResults[5][9];
    double result = 0.0;
    int position = 0;
    /* Initial parameters (powers of coefficients), which can be changed */
    double b0_input = -14.0;
    double b1_input = -3.0;
    double b2_input = -10.0;
    double b3_input = -5.0;
    double b4_input = -20.0;
    double b5_input = -20.0;
    double b6_input = -21.0;
    double b7_input = -30.0;
    int b0_index = 0;
    int b1_index = 0;
    int b2_index = 0;
    int b3_index = 0;
    int b4_index = 0;
    int b5_index = 0;
    int b6_index = 0;
    int b7_index = 0;
    int repeats;
    double positionPrev;
    for (int iFunc=0; iFunc < 5; ++iFunc) {
        //if (iFunc == 2) continue;     // this line may be uncommented to skip the longest fitted function when testing the library
        printf("I started fitting the mortality data to %s\n", (iFunc == 0 ? "Erf-derived function" :
                                                       (iFunc == 1 ? "Logistic-derived function" :
                                                        (iFunc == 2 ? "Gudermannian-derived function\nFitting this function takes times longer than either of the previous two functions,\nso please be patient" :
                                                         (iFunc == 3 ? "Algebraic function derived from x over sqrt(1 + x^2)" :
                                                          "Algebraic function derived from x over (1 + abs(x))"
                                                          )
                                                         )
                                                        )
                                                       )
               );
        fflush(stdout);
        for (int iPrecision=0; iPrecision < 5; ++iPrecision) {
            printf("\tFitting with precision %.4f\n", pow(10, -iPrecision));
            fflush(stdout);
            repeats = 0;
            position = 0;
            while (position != 3280) {
                if (repeats > 25 && iPrecision > 0) {
                    --iPrecision;
                    printf("\tFitting with precision %.4f again because slope ascending is too slow\n", pow(10, -iPrecision));
                    fflush(stdout);
                }
                positionPrev = position;
                position = oneStep(iFunc, &result, ages, the_outcomes, length, b0_input, b1_input, b2_input, b3_input, b4_input, b5_input, b6_input, b7_input, iPrecision);
                b0_index = position / 2187;
                b1_index = position % 2187 / 729;
                b2_index = position % 729 / 243;
                b3_index = position % 243 / 81;
                b4_index = position % 81 / 27;
                b5_index = position % 27 / 9;
                b6_index = position % 9 / 3;
                b7_index = position % 3;
                b0_input += (b0_index - 1) * pow(10.0, -iPrecision);
                b1_input += (b1_index - 1) * pow(10.0, -iPrecision);
                b2_input += (b2_index - 1) * pow(10.0, -iPrecision);
                b3_input += (b3_index - 1) * pow(10.0, -iPrecision);
                b4_input += (b4_index - 1) * pow(10.0, -iPrecision);
                b5_input += (b5_index - 1) * pow(10.0, -iPrecision);
                b6_input += (b6_index - 1) * pow(10.0, -iPrecision);
                b7_input += (b7_index - 1) * pow(10.0, -iPrecision);
                if (positionPrev == position) ++repeats;
            }
        }
        finalResults[iFunc][0] = b0_input;
        finalResults[iFunc][1] = b1_input;
        finalResults[iFunc][2] = b2_input;
        finalResults[iFunc][3] = b3_input;
        finalResults[iFunc][4] = b4_input;
        finalResults[iFunc][5] = b5_input;
        finalResults[iFunc][6] = b6_input;
        finalResults[iFunc][7] = b7_input;
        finalResults[iFunc][8] = result;
    }
    //finalResults[2][8] = -150.0;    // this line can be uncommented to arbitrarily force any of the five functions with their best fitted parameters be be reported to Python script for plotting and saving their fitted formulas
    
    /* the output below help compare the five functions in terms of their fit to the data */
    printf("\nFunction 1 erf parameters:\t\t\tML estimate\t%.16f\n\t%e + %ex + %ex^2 + %ex^3 + %ex^4 + %ex^5 + %ex^6 + %ex^7\n", finalResults[0][8],
           pow(10.0, finalResults[0][0]),
           pow(10.0, finalResults[0][1]),
           pow(10.0, finalResults[0][2]),
           pow(10.0, finalResults[0][3]),
           pow(10.0, finalResults[0][4]),
           pow(10.0, finalResults[0][5]),
           pow(10.0, finalResults[0][6]),
           pow(10.0, finalResults[0][7]));
    printf("\nFunction 2 tanh parameters:\t\tML estimate\t%.16f\n\t%e + %ex + %ex^2 + %ex^3 + %ex^4 + %ex^5 + %ex^6 + %ex^7\n", finalResults[1][8],
           pow(10.0, finalResults[1][0]),
           pow(10.0, finalResults[1][1]),
           pow(10.0, finalResults[1][2]),
           pow(10.0, finalResults[1][3]),
           pow(10.0, finalResults[1][4]),
           pow(10.0, finalResults[1][5]),
           pow(10.0, finalResults[1][6]),
           pow(10.0, finalResults[1][7]));
    printf("\nFunction 3 GudFunc parameters:\t\tML estimate\t%.16f\n\t%e + %ex + %ex^2 + %ex^3 + %ex^4 + %ex^5 + %ex^6 + %ex^7\n", finalResults[2][8],
           pow(10.0, finalResults[2][0]),
           pow(10.0, finalResults[2][1]),
           pow(10.0, finalResults[2][2]),
           pow(10.0, finalResults[2][3]),
           pow(10.0, finalResults[2][4]),
           pow(10.0, finalResults[2][5]),
           pow(10.0, finalResults[2][6]),
           pow(10.0, finalResults[2][7]));
    printf("\nFunction 4 x over x2 parameters:\t\tML estimate\t%.16f\n\t%e + %ex + %ex^2 + %ex^3 + %ex^4 + %ex^5 + %ex^6 + %ex^7\n", finalResults[3][8],
           pow(10.0, finalResults[3][0]),
           pow(10.0, finalResults[3][1]),
           pow(10.0, finalResults[3][2]),
           pow(10.0, finalResults[3][3]),
           pow(10.0, finalResults[3][4]),
           pow(10.0, finalResults[3][5]),
           pow(10.0, finalResults[3][6]),
           pow(10.0, finalResults[3][7]));
    printf("\nFunction 5 x over abs x parameters:\tML estimate\t%.16f\n\t%e + %ex + %ex^2 + %ex^3 + %ex^4 + %ex^5 + %ex^6 + %ex^7\n\n", finalResults[4][8],
           pow(10.0, finalResults[4][0]),
           pow(10.0, finalResults[4][1]),
           pow(10.0, finalResults[4][2]),
           pow(10.0, finalResults[4][3]),
           pow(10.0, finalResults[4][4]),
           pow(10.0, finalResults[4][5]),
           pow(10.0, finalResults[4][6]),
           pow(10.0, finalResults[4][7]));
    fflush(stdout);
    int resulting = 0;
    int mainresulting = 0;
    for (int i=1; i < 5; ++i)
        if (finalResults[i][8] > finalResults[resulting][8])
            resulting = i;
    for (int i=0; i < 9; ++i) output[i] = finalResults[resulting][i];
    mainresulting = resulting;
    /* since the Gudermannian-derived function used here can be non-monotonic, whenever it seems the best fit, the second best fit is reported as well */
    if (resulting == 2) {
        resulting = 0;
        for (int i=1; i < 5; ++i) {
            if (i == 2) continue;
            if (finalResults[i][8] > finalResults[resulting][8])
                resulting = i;
        }
        for (int i=0; i < 9; ++i) output[9 + i] = finalResults[resulting][i];
        *secondRes = resulting;
    }
    return mainresulting;
}
