/*
This C code and the code of the affiliated Python scripts may help
you fit analytically expressed functions to the COVID-19 mortality
data to model age-adjusted mortality risk using maximum likelihood
point estimates.

This software was desinged in two parts: a shared C library to
dramatically speed up the calculation and a Python wrapper script
to feed the input data into the affiliated shared C library and
finally plot the fitted curves. A sample Python script.py file,
which helps more easily import and transform the input tabular data
and imports and communicates with the Python wrapper module, is also
attached.

The shared C library provides the opportunity to test arbitrary
functions on condition that, in the domain between 0 and 120+, they
return values between 0.0 and 1.0 - otherwise, in this scenario, it
will make no sense.

The Python wrapper interface function "fitFunctionWrapper" accepts
a two-column pandas DataFrame with:
- the first column 'age' of the numpy numerical data type, e.g.,
  numpy.float64 or numpy.intc (the float datatype allows to accomodate
  data that specify full dates of birth instead of years of birth),
- the second column 'outcome' of the numpy numerical data type, e.g.,
  numpy.intc, where non-zero (e.g., 1) means death and zero means a
  more positive outcome.

It return a tuple of two objects of the class "bestFit" defined in the
same wrapper module. The first object contains the calculated
parameters and the number of the best fitted function.

The attached script.py sample can be modified to supply case-by-case
data I don't yet have access to or have failed to find.

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

/* The macro below was created instead of a function to avoid function call overhead */
#define internalLogL(x, signs, b0, b1, b2, b3, b4, b5, b6, b7) (pow(-1.0, (double) (signs & (unsigned char) 0b00000001)) * b0 + pow(-1.0, (double) ((signs & (unsigned char) 0b00000010) >> 1)) * b1 * x + pow(-1.0, (double) ((signs & (unsigned char) 0b00000100) >> 2)) * b2 * pow(x, 2.0) + pow(-1.0, (double) ((signs & (unsigned char) 0b00001000) >> 3)) * b3 * pow(x, 3.0) + pow(-1.0, (double) ((signs & (unsigned char) 0b00010000) >> 4)) * b4 * pow(x, 4.0) + pow(-1.0, (double) ((signs & (unsigned char) 0b00100000) >> 5)) * b5 * pow(x, 5.0) + pow(-1.0, (double) ((signs & (unsigned char) 0b01000000) >> 6)) * b6 * pow(x, 6.0) + pow(-1.0, (double) ((signs & (unsigned char) 0b10000000) >> 7)) * b7 * pow(x, 7.0))
#define internalLogS(x, signs, b0, b1, b2, b3, b4, b5) (pow(-1.0, (double) (signs & (unsigned char) 0b00000001)) * b0 + pow(-1.0, (double) ((signs & (unsigned char) 0b00000010) >> 1)) * b1 * x + pow(-1.0, (double) ((signs & (unsigned char) 0b00000100) >> 2)) * b2 * pow(x, 2.0) + pow(-1.0, (double) ((signs & (unsigned char) 0b00001000) >> 3)) * b3 * pow(x, 3.0) + pow(-1.0, (double) ((signs & (unsigned char) 0b00010000) >> 4)) * b4 * pow(x, 4.0) + pow(-1.0, (double) ((signs & (unsigned char) 0b00100000) >> 5)) * b5 * pow(x, 5.0))
#define logVerified(x) ( (x <= 0.0) && (x >= 1.0) ? -DBL_MAX : log(x) )
#define indexConverter(x) (1 + (x > 0 ? (int)pow(3,1) : 0) + (x > 1 ? (int)pow(3,2) : 0) + (x > 2 ? (int)pow(3,3) : 0) + (x > 3 ? (int)pow(3,4) : 0) + (x > 4 ? (int)pow(3,5) : 0) + (x > 5 ? (int)pow(3,6) : 0) + (x > 6 ? (int)pow(3,7) : 0))
#define THREADS_MAX 6561   // 3 ^ 8 — the former is the number of tests for each parameter per step, the latter is the number of fitted parameters. So many threads don't significantly impede performance in practice (though you may want to reassess that) but will use whatever number of CPU cores and threads your laptop, workstation, or server has.
#define START_FUNCTION 1   // this can be used to "hardcode" to fit fewer functions than added to this code
#define STOP_FUNCTION 10   // this can be used to "hardcode" to fit fewer functions than added to this code
#define TOTAL_NUMBER_OF_FUNCTIONS 10

/* these variables were made global to make them accessible (read-only) to all threads */
char signString[9];
unsigned char signs;
static int func_g, * outcome_g, length_g, precision_g;
static double b0_g, b1_g, b2_g, b3_g, b4_g, b5_g, b6_g, b7_g, * age_g, * result_g;

/* The ten fitted functions */

static char funcName0[] = "Erf-derived function";
static double erfLog(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7){
    double result = internalLogL(x, signs, b0, b1, b2, b3, b4, b5, b6, b7);
    if (result <= 0.0) return -DBL_MAX;
    result = erf(log(result)) * 0.5 + 0.5;
    return outcome ? logVerified(result) : logVerified(1.0 - result);
}

static char funcName1[] = "Erf-derived function with floor and ceiling";
static double erfLogFC(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7){
    double result = internalLogS(x, signs, b0, b1, b2, b3, b4, b5);
    if (result <= 0.0) return -DBL_MAX;
    result = erf(log(result)) * (0.5 - b7) + 0.5 - b7 + b6;
    return outcome ? logVerified(result) : logVerified(1.0 - result);
}

static char funcName2[] = "Logistic-derived function";
static double hyperbTan(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7){
    double result = tanh(internalLogL(x, signs, b0, b1, b2, b3, b4, b5, b6, b7)) * 0.5 + 0.5;
    return outcome ? logVerified(result) : logVerified(1.0 - result);
}

static char funcName3[] = "Logistic-derived function with floor and ceiling";
static double hyperbTanFC(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7){
    double result = tanh(internalLogS(x, signs, b0, b1, b2, b3, b4, b5)) * (0.5 - b7) + 0.5 - b7 + b6;
    return outcome ? logVerified(result) : logVerified(1.0 - result);
}

#define LONG_FUNC_NUMBER 4
static char funcName4[] = "Gudermannian-derived function";
static double GudFunc(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7){
    double result = atan(tanh(internalLogL(x, signs, b0, b1, b2, b3, b4, b5, b6, b7))) * M_1_PI * 2.0 + 0.5;
    return outcome ? logVerified(result) : logVerified(1.0 - result);
}

static char funcName5[] = "Gudermannian-derived function with floor and ceiling";
static double GudFuncFC(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7){
    double result = atan(tanh(internalLogS(x, signs, b0, b1, b2, b3, b4, b5))) * M_1_PI * 4.0 * (0.5 - b7) + 0.5 - b7 + b6;
    return outcome ? logVerified(result) : logVerified(1.0 - result);
}

static char funcName6[] = "Algebraic function derived from x over sqrt(1 + x^2)";
static double xOverX2(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7){
    double temp = internalLogL(x, signs, b0, b1, b2, b3, b4, b5, b6, b7);
    double result = temp * pow(1.0 + pow(temp, 2.0), -0.5) * 0.5 + 0.5;
    return outcome ? logVerified(result) : logVerified(1.0 - result);
}

static char funcName7[] = "Algebraic function derived from x over sqrt(1 + x^2) with floor and ceiling";
static double xOverX2FC(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7){
    double temp = internalLogS(x, signs, b0, b1, b2, b3, b4, b5);
    double result = temp * pow(1.0 + pow(temp, 2.0), -0.5) * (0.5 - b7) + 0.5 - b7 + b6;
    return outcome ? logVerified(result) : logVerified(1.0 - result);
}

static char funcName8[] = "Algebraic function derived from x over (1 + abs(x))";
static double xOverAbs(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7){
    double temp = internalLogL(x, signs, b0, b1, b2, b3, b4, b5, b6, b7);
    double result = temp / (1 +fabs(temp)) * 0.5 + 0.5;
    return outcome ? logVerified(result) : logVerified(1.0 - result);
}

static char funcName9[] = "Algebraic function derived from x over (1 + abs(x)) with floor and ceiling";
static double xOverAbsFC(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7){
    double temp = internalLogS(x, signs, b0, b1, b2, b3, b4, b5);
    double result = temp / (1 +fabs(temp)) * (0.5 - b7) + 0.5 - b7 + b6;
    return outcome ? logVerified(result) : logVerified(1.0 - result);
}

/* array of function pointers to facilitate their calls by numbers */
static double (*testFunc[TOTAL_NUMBER_OF_FUNCTIONS])(double x, int outcome, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7);

static char * funcNames[TOTAL_NUMBER_OF_FUNCTIONS];

static void convertSigns() {
    for (int i=0; i < 8; ++i)
        signString[i] = (signs & (unsigned char) pow(2, i)) ? '-' : '+';
}

static void * getML(void * threadId) {
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

static int oneStep(int func, double * result, double * age, int * outcome, int length, double b0, double b1, double b2, double b3, double b4, double b5, double b6, double b7, int precision) {
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

/* the function that needs to be called from the Python (wrapper) script */
int fitFunction(double * ages, int * the_outcomes, int length, double * output, int * secondRes, int * sign1, int * sign2, int * functionsToTest) {
    FILE * fp;   // file pointer for the stop signal
    signString[8] = '\0';
    testFunc[0] = &erfLog;
    testFunc[1] = &erfLogFC;
    testFunc[2] = &hyperbTan;
    testFunc[3] = &hyperbTanFC;
    testFunc[4] = &GudFunc;
    testFunc[5] = &GudFuncFC;
    testFunc[6] = &xOverX2;
    testFunc[7] = &xOverX2FC;
    testFunc[8] = &xOverAbs;
    testFunc[9] = &xOverAbsFC;
    funcNames[0] = funcName0;
    funcNames[1] = funcName1;
    funcNames[2] = funcName2;
    funcNames[3] = funcName3;
    funcNames[4] = funcName4;
    funcNames[5] = funcName5;
    funcNames[6] = funcName6;
    funcNames[7] = funcName7;
    funcNames[8] = funcName8;
    funcNames[9] = funcName9;
    double finalResults[STOP_FUNCTION][9];
    unsigned char finalSigns[STOP_FUNCTION];
    double result = 0.0;
    double resultPrev = 0.0;
    int position = 0;
    /* Initial parameters (powers of coefficients), which can be changed.
    Both the speed of fitting and the local maximum where you're gonna get stuck highly depend on the choice of these initial parameters.
    Play with them to get better fitting. */
    double b0_input_seed = -13.7895;
    double b1_input_seed = -2.5031;
    double b2_input_seed = -4.3979;
    double b3_input_seed = -7.7828;
    double b4_input_seed = -18.3195;
    double b5_input_seed = -19.7568;
    double b6_input_seed = -26.0665;
    double b7_input_seed = -31.0665;
    /* double b0_input_seed = -13.9234;
    double b1_input_seed = -2.5907;
    double b2_input_seed = -4.4888;
    double b3_input_seed = -18.2855;
    double b4_input_seed = -22.5474;
    double b5_input_seed = -78.2407;
    double b6_input_seed = -82.5504;
    double b7_input_seed = -87.5504; */
    double b0_input, b1_input, b2_input, b3_input, b4_input, b5_input, b6_input, b7_input;
    int b0_index = 0;
    int b1_index = 0;
    int b2_index = 0;
    int b3_index = 0;
    int b4_index = 0;
    int b5_index = 0;
    int b6_index = 0;
    int b7_index = 0;
    int repeats;
    int repeatsWarning = 0;
    double positionPrev;
    double tempCoefficient;
    double tempML;
    int firstFunction = 1;
    int resulting = 0;
    for (int iFunc = START_FUNCTION - 1; iFunc < STOP_FUNCTION; ++iFunc) {
        if (!functionsToTest[iFunc]) continue;
        if (firstFunction) resulting = iFunc;
        signs = (unsigned char) * sign1;
        while(1) {
            convertSigns();
            b0_input = b0_input_seed;
            b1_input = b1_input_seed;
            b2_input = b2_input_seed;
            b3_input = b3_input_seed;
            b4_input = b4_input_seed;
            b5_input = b5_input_seed;
            b6_input = b6_input_seed;
            b7_input = b7_input_seed;
            repeatsWarning = 0;
            result = 0.0;
            resultPrev = 0.0;
            printf("I started fitting the mortality data to %s with signs x%02x %s\n", funcNames[iFunc], signs, signString);
            fflush(stdout);
            for (int iPrecision=0; iPrecision < 5; ++iPrecision) {
                ++repeatsWarning;
                printf("\tFitting with precision %.4f\n", pow(10, -iPrecision));
                if (repeatsWarning % 20 == 19) {
                    printf("***********************************************************************************\n\t\tIf you start to suspect that your computer got into a dead loop\n\t\t— Nope, the ML estimate is still increasing:\n\t\t\tit is %14.10f now\n", result);
                    if (resultPrev) printf("\t\t\t  vs. %14.10f, which was 20 lines above\n", resultPrev);
                    puts("***********************************************************************************");
                    resultPrev = result;
                }
                fflush(stdout);
                repeats = 0;
                position = 0;
                while (position != 3280) {
                    if (repeats > 25 && iPrecision > 0) {
                        ++repeatsWarning;
                        --iPrecision;
                        printf("\tFitting with precision %.4f again because slope ascending is too slow\n", pow(10, -iPrecision));
                        if (repeatsWarning % 20 == 19) {
                            printf("***********************************************************************************\n\t\tIf you start to suspect that your computer got into a dead loop\n\t\t— Nope, the ML estimate is still increasing:\n\t\t\tit is %14.10f now\n", result);
                            if (resultPrev) printf("\t\t\t  vs. %14.10f, which was 20 lines above\n", resultPrev);
                            puts("***********************************************************************************");
                            resultPrev = result;
                        }
                        fflush(stdout);
                    }
                    positionPrev = position;
                    position = oneStep(iFunc, &result, ages, the_outcomes, length, b0_input, b1_input, b2_input, b3_input, b4_input, b5_input, b6_input, b7_input, iPrecision);
                    //printf("%15.10f\t", result);      // this may be uncommented to print each ML estimate along the way
                    fflush(stdout);
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
                    if ((fp = fopen("stop.txt", "r")) != NULL) {
                        fclose(fp);
                        break;
                    }
                }
                if ((fp = fopen("stop.txt", "r")) != NULL) {
                    fclose(fp);
                    remove("stop.txt");
                    puts("I have received the signal to stop. The calculation has stoped. You will see the intermediate results.");
                    fflush(stdout);
                    break;
                }
            }
            if (signs == (unsigned char) * sign1 || result > finalResults[iFunc][8]) {
                finalResults[iFunc][0] = pow(10.0, b0_input);
                finalResults[iFunc][1] = pow(10.0, b1_input);
                finalResults[iFunc][2] = pow(10.0, b2_input);
                finalResults[iFunc][3] = pow(10.0, b3_input);
                finalResults[iFunc][4] = pow(10.0, b4_input);
                finalResults[iFunc][5] = pow(10.0, b5_input);
                finalResults[iFunc][6] = pow(10.0, b6_input);
                finalResults[iFunc][7] = pow(10.0, b7_input);
                finalResults[iFunc][8] = result;
                finalSigns[iFunc] = signs;
                for (int i=0; i < 8; ++i) {
                    tempCoefficient = finalResults[iFunc][i];
                    tempML = testFunc[func_g](130.0, 1, finalResults[iFunc][0],
                                                        finalResults[iFunc][1],
                                                        finalResults[iFunc][2],
                                                        finalResults[iFunc][3],
                                                        finalResults[iFunc][4],
                                                        finalResults[iFunc][5],
                                                        finalResults[iFunc][6],
                                                        finalResults[iFunc][7]);
                    finalResults[iFunc][i] = 0.0;
                    if (tempML != testFunc[func_g](130.0, 1, finalResults[iFunc][0],
                                                             finalResults[iFunc][1],
                                                             finalResults[iFunc][2],
                                                             finalResults[iFunc][3],
                                                             finalResults[iFunc][4],
                                                             finalResults[iFunc][5],
                                                             finalResults[iFunc][6],
                                                             finalResults[iFunc][7]))
                        finalResults[iFunc][i] = tempCoefficient;
                }
                if(finalResults[iFunc][0]) finalResults[iFunc][0] *= pow(-1.0, (double) (signs & (unsigned char) 0b00000001));
                if(finalResults[iFunc][1]) finalResults[iFunc][1] *= pow(-1.0, (double) ((signs & (unsigned char) 0b00000010) >> 1));
                if(finalResults[iFunc][2]) finalResults[iFunc][2] *= pow(-1.0, (double) ((signs & (unsigned char) 0b00000100) >> 2));
                if(finalResults[iFunc][3]) finalResults[iFunc][3] *= pow(-1.0, (double) ((signs & (unsigned char) 0b00001000) >> 3));
                if(finalResults[iFunc][4]) finalResults[iFunc][4] *= pow(-1.0, (double) ((signs & (unsigned char) 0b00010000) >> 4));
                if(finalResults[iFunc][5]) finalResults[iFunc][5] *= pow(-1.0, (double) ((signs & (unsigned char) 0b00100000) >> 5));
                if(finalResults[iFunc][6]) finalResults[iFunc][6] *= pow(-1.0, (double) ((signs & (unsigned char) 0b01000000) >> 6));
                if(finalResults[iFunc][7]) finalResults[iFunc][7] *= pow(-1.0, (double) ((signs & (unsigned char) 0b10000000) >> 7));
            }
            printf("\t\tML is %20.16f\n", result);
            fflush(stdout);
            if (signs == (* sign1) + 2 || * sign2) break;
            if (iFunc % 2 && signs / (unsigned char) 0b01000000) break;
            ++signs;
        }
    }
    //finalResults[2][8] = -150.0;    // this line can be uncommented to arbitrarily force any of the five functions with their best fitted parameters be be reported to Python script for plotting and saving their fitted formulas
    
    /* the output below help compare the ten functions in terms of their fit to the data */
    for (int iFunc = START_FUNCTION - 1; iFunc < STOP_FUNCTION; ++iFunc) {
        if (!functionsToTest[iFunc]) continue;
        printf("\nFunction %i:\t\t%s\n\tML estimate:\t%.16f\n\tParameters:\t%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n\tSigns:\t\tx%02x\t%s\n",
               iFunc,
               funcNames[iFunc],
               finalResults[iFunc][8],
               finalResults[iFunc][0],
               finalResults[iFunc][1],
               finalResults[iFunc][2],
               finalResults[iFunc][3],
               finalResults[iFunc][4],
               finalResults[iFunc][5],
               finalResults[iFunc][6],
               finalResults[iFunc][7],
               finalSigns[iFunc],
               signString);
    }
    fflush(stdout);
    for (int iFunc = START_FUNCTION - 1; iFunc < STOP_FUNCTION; ++iFunc) {
        if (!functionsToTest[iFunc]) continue;
        if (iFunc == resulting) continue;
        if (finalResults[iFunc][8] > finalResults[resulting][8])
            resulting = iFunc;
    }
    for (int i=0; i < 9; ++i) output[i] = finalResults[resulting][i];
    *sign1 = (int) finalSigns[resulting];
    return resulting;
}
