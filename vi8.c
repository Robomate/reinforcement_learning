#include<stdio.h>
#include<stdlib.h>
#include<string.h>
/**
 * RL Algos: Value Iteration 
 * Gridworld Example
 */ 

////////////////////////   
// init globals
////////////////////////   
#define ITERATIONS 100 // number of iterations
#define DISCOUNT 0.9 // discount factor
const double trans[3] = {0.1,0.8,0.1}; // transistion probs
const char *actions[] = {"up   ", "right", "down ", "left "}; // actions
#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr)[0])) // macro for size of array
#define LENVEC(x)  (sizeof(x) / sizeof((x)[0]))
// init size of grid world
#define ROWS 3 // number of rows
#define COLS 4 // number of columns
#define RV 4 // special cases
#define CV 3 // special cases
#define STRING_LEN 10 // string length
// init tuple for maximum
struct Max_tuple 
{
 double max_val; 
 int max_idx;   
}; 
// init special cases as struct (to do list!!, not used)
struct VAL
{
 double V1 [RV][CV]; // corner case, up-left
 double V2 [RV][CV];
 double V3 [RV][CV];
 double V4 [RV][CV];
 double V5 [RV][CV];
 double V6 [RV][CV];
 double V7 [RV][CV];
 double V8 [RV][CV];
 double V9 [RV][CV];
};

////////////////////////                     
// init functions
////////////////////////
void fill_mat_const( double array[][COLS], double val)
{
    /* fills a matrix with a constant value */
    int i;
    int j;
    for(i=0; i < ROWS; i++)
    {
     for(j=0; j<COLS; ++j)
     {
      array[i][j] = val;
     }   
    }
}

void print_mat( double array[][COLS])
{
    /* prints out a double matrix */
    int i,j;
    for(i=0; i < ROWS; i++)
    {
     for(j=0; j<COLS; ++j)
     {
      printf(" %3.2lf", array[i][j]);
     }
     printf("\n");   
    }
printf("\n");
}

void print_mat_V1( double array[][CV])
{
    /* prints out a double matrix */
    int i,j;
    for(i=0; i < RV; i++)
    {
     for(j=0; j<CV; ++j)
     {
      printf(" %3.2lf", array[i][j]);
     }
     printf("\n");   
    }
printf("\n");
}

void print_char_mat( char array[][COLS][STRING_LEN])
{
    /* prints out a char table */
    int i,j;
    for(i=0; i < ROWS; i++)
    {
     for(j=0; j<COLS; ++j)
     {
      printf(" %s", array[i][j]);
     }
     printf("\n");   
    }
printf("\n");
}

 void fill_vec_zero( double vec[])
{
    int i;
    for( i = 0; i < RV; i++)
    {
        vec[i] = 0;
    }
}

 void print_vec(double vec[])
{   
    int i;
    printf("vector elements :\n");
    for (i = 0; i < RV; ++i)
    { 
        printf(" %3.2lf\n", vec[i]);
    }
}

void matvec_mul( double mat[][CV], const double vec[], double out[])
{
  /* matrix vector multiplicaton */
  int r, c;
  fill_vec_zero(out); // init with zeros
  for (r=0; r<RV; r++)
  {
    for (c=0; c<CV; c++)
    {
      out[r] += mat[r][c] * vec[c];
      //printf(" %3.2lf\n", out[r]);
    }
    //printf("\n");
  }
  //print_vec(out);
}

struct Max_tuple find_max_vec(double array[])
 {
     /* find maximum value and index in matrix */
      int r; 
      int idx = 0;
      double maximum;
      struct Max_tuple max; // struct instance
      maximum = array[0];
   for( r = 0 ; r < RV ; r++ )
   {
     if ( array[r] > maximum )
         {
            maximum = array[r];
            idx = r;
         }  
   }
   // assign values to struct
   max.max_val = maximum;
   max.max_idx = idx;
   return max;
 }

// init main
int main(int argc, char** argv)
{
  double R[ROWS][COLS]; // reward table
  double V[ROWS][COLS]; // value table
  double V0[ROWS][COLS]; // value table at t-1
  char P[ROWS][COLS][STRING_LEN]; // value table
  double VZ[RV][CV]; // special cases
  double M[RV]; // matmul result
  struct Max_tuple max; // max struct instance
  int i,r,c;
  system("clear"); // clear screen
  
  // fill tables with zeros
  fill_mat_const(R,0);
  R[0][3] = 1;
  R[1][3] = -1;
  printf("\nInitial Reward Table\n");
  print_mat(R);
  fill_mat_const(V,0);
  printf("Initial Value Table\n");
  print_mat(V);

  // start iterations
  for(i=1; i < ITERATIONS+1; i++)
  {
    //printf("Iterations: %d\n",i);
    memcpy(V0,V, ROWS*COLS*sizeof(double)); // copy V to V0
    // start recursion
    for(r=0; r < ROWS; r++)
    {
     for(c=0; c<COLS; c++)
     {
        // 4 corner cases first, then 4 rim cases, else standard case
        if( r == 0 && c == 0 ) 
        {
           double V1 [RV][CV] = {
                        { V0[r][c]  , V0[r][c]  , V0[r][c+1] },
                        { V0[r][c]  , V0[r][c+1], V0[r+1][c] },
                        { V0[r][c+1], V0[r+1][c], V0[r][c]   }, 
                        { V0[r+1][c], V0[r][c]  , V0[r][c]   }
                    };
            memcpy(VZ,V1, ROWS*COLS*sizeof(double)); // copy V1 to VZ
        }
        else if( r == 0 && c == COLS-1 )
        {
            // corner case, up-right
             double V1 [RV][CV] = {
                        { V0[r][c-1], V0[r][c]  , V0[r][c]   },
                        { V0[r][c]  , V0[r][c]  , V0[r+1][c] },
                        { V0[r][c]  , V0[r+1][c], V0[r][c-1] }, 
                        { V0[r+1][c], V0[r][c-1], V0[r][c]   }
                    };
            memcpy(VZ,V1, ROWS*COLS*sizeof(double)); // copy V1 to VZ
            //printf(" VZ\n");
            //print_mat_V1(V1);
            //print_mat_V1(VZ);    
        }
          else if( r == ROWS-1 && c == COLS-1 )
        {
            // corner case, down-right
             double V1 [RV][CV] = {
                        { V0[r][c-1], V0[r-1][c], V0[r][c]   },
                        { V0[r-1][c], V0[r][c]  , V0[r][c]   },
                        { V0[r][c]  , V0[r][c]  , V0[r][c-1] }, 
                        { V0[r][c]  , V0[r][c-1], V0[r-1][c] }
                    };
            memcpy(VZ,V1, ROWS*COLS*sizeof(double)); // copy V1 to VZ   
        }
          else if( r == ROWS-1 && c == 0)
        {
            // corner case, down-left
             double V1 [RV][CV] = {
                        { V0[r][c]  , V0[r-1][c], V0[r][c+1] },
                        { V0[r-1][c], V0[r][c+1], V0[r][c]   },
                        { V0[r][c+1], V0[r][c]  , V0[r][c]   }, 
                        { V0[r][c]  , V0[r][c]  , V0[r-1][c] }
                    };
            memcpy(VZ,V1, ROWS*COLS*sizeof(double)); // copy V1 to VZ    
        }
          else if( r == 0 && c != 0 && c != COLS-1 )
        {
            // rim case, up
             double V1 [RV][CV] = {
                        { V0[r][c-1], V0[r][c]  , V0[r][c+1] },
                        { V0[r][c]  , V0[r][c+1], V0[r+1][c] },
                        { V0[r][c+1], V0[r+1][c], V0[r][c-1] }, 
                        { V0[r+1][c], V0[r][c-1], V0[r][c]   }
                    };
            memcpy(VZ,V1, ROWS*COLS*sizeof(double)); // copy V1 to VZ     
        }
          else if( c == COLS-1 && r != 0 && r != ROWS-1 )
        {
            //  rim case, right
             double V1 [RV][CV] = {
                        { V0[r][c-1], V0[r-1][c], V0[r][c]   },
                        { V0[r-1][c], V0[r][c]  , V0[r+1][c] },
                        { V0[r][c]  , V0[r+1][c], V0[r][c-1] }, 
                        { V0[r+1][c], V0[r][c-1], V0[r-1][c] }
                    };
            memcpy(VZ,V1, ROWS*COLS*sizeof(double)); // copy V1 to VZ    
        }
          else if( r == ROWS-1 && c != 0 && c != COLS-1 )
        {
            //  rim case, down
             double V1 [RV][CV] = {
                        { V0[r][c-1], V0[r-1][c], V0[r][c+1] },
                        { V0[r-1][c], V0[r][c+1], V0[r][c]   },
                        { V0[r][c+1], V0[r][c]  , V0[r][c-1] }, 
                        { V0[r][c]  , V0[r][c-1], V0[r-1][c] }
                    };
            memcpy(VZ,V1, ROWS*COLS*sizeof(double)); // copy V1 to VZ 
        }
          else if( c == 0 && r != 0 && r != ROWS-1 )
        {
            //  rim case, left
             double V1 [RV][CV] = {
                        { V0[r][c]  , V0[r-1][c], V0[r][c+1] },
                        { V0[r-1][c], V0[r][c+1], V0[r+1][c] },
                        { V0[r][c+1], V0[r+1][c], V0[r][c]   }, 
                        { V0[r+1][c], V0[r][c]  , V0[r-1][c] }
                    }; 
            memcpy(VZ,V1, ROWS*COLS*sizeof(double)); // copy V1 to VZ    
        }
        else 
        {
            // standard case, action order: (North, East, South, West)
            double V1 [RV][CV] = {
                        { V0[r][c-1], V0[r-1][c], V0[r][c+1] },
                        { V0[r-1][c], V0[r][c+1], V0[r+1][c] },
                        { V0[r][c+1], V0[r+1][c], V0[r][c-1] }, 
                        { V0[r+1][c], V0[r][c-1], V0[r-1][c] }
                    };
            memcpy(VZ,V1, ROWS*COLS*sizeof(double)); // copy V1 to VZ
        }
        // apply Bellman Update
        matvec_mul(VZ, trans, M); // matmul
        max = find_max_vec(M); // get best action
        //printf("Max Value: %3.2lf\n", max.max_val);
        //printf("Max Index: %d\n", max.max_idx);
        V[r][c] = R[r][c] + DISCOUNT * max.max_val; // Update Value Table
        strcpy(P[r][c], actions[max.max_idx]); // Update Policy Table
        //printf("Action: %s\n", P[0][0]);    
     }
   }
  // Print out results:
  //printf(" Value Table\n");
  //print_mat(V);
  //printf(" Policy Table\n");
  //print_char_mat(P);
  }
  // Print out final results:
  printf("Iterations: %d\n",i-1);
  printf(" Value Table t-1\n");
  print_mat(V0);
  printf(" Value Table\n");
  print_mat(V);
  printf(" Policy Table\n");
  print_char_mat(P);
  return 0;
}



// cases
/* 
V0[r][c]
V0[r][c+1]
V0[r+1][c]
V0[r+1][c+1]
V0[r][c-1]
V0[r-1][c]
V0[r-1][c-1]
 */