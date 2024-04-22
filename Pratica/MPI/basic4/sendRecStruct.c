#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
        { int a;
          double b;
          char c;
        } XYZ;

int main (int argc, char *argv[])
{
   int rank, size;
   XYZ sndData = {13, 0.5, 'a'},
       recData;

   MPI_Init (&argc, &argv);
   MPI_Comm_rank (MPI_COMM_WORLD, &rank);
   MPI_Comm_size (MPI_COMM_WORLD, &size);
   if (size < 2)
      { if (rank == 0) printf ("Too few processes!\n");
        MPI_Finalize ();
        return EXIT_FAILURE;
      }
   if (rank == 0)
      { printf ("Transmitted message: %d - %.3f - %c \n", sndData.a, sndData.b, sndData.c);
        MPI_Send ((char *) &sndData, sizeof (XYZ), MPI_BYTE, 1, 0, MPI_COMM_WORLD);
      }
      else if (rank == 1)
              { MPI_Recv ((char *) &recData, sizeof (XYZ), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf ("Received message: %d - %.3f - %c \n", recData.a, recData.b, recData.c);
              }
   MPI_Finalize ();
   return EXIT_SUCCESS;
}
