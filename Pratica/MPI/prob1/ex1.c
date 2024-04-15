#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main (int argc, char *argv[])
{
   int rank, size;
   char data[] = "I am here!",
        *recData;

   MPI_Init (&argc, &argv);
   MPI_Comm_rank (MPI_COMM_WORLD, &rank);
   MPI_Comm_size (MPI_COMM_WORLD, &size);

    if (size == 1)
    {
        MPI_Request request;
        printf ("Process %d Transmitted message: %s \n", rank, data);
        MPI_Isend (data, strlen (data), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &request);
    } else {
        if (rank != size-1)
        {
            printf ("Process %d Transmitted message: %s \n", rank, data);
            MPI_Send (data, strlen (data), MPI_CHAR, rank+1, 0, MPI_COMM_WORLD);
        } else {
            printf ("Process %d Transmitted message: %s \n", rank, data);
            MPI_Send (data, strlen (data), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }

    int i;
    recData = malloc (100);
    for (i = 0; i < 100; i++)
        recData[i] = '\0';
    MPI_Recv (recData, 100, MPI_CHAR, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf ("Process %d Received message: %s \n",rank, recData);


   MPI_Finalize ();

   return EXIT_SUCCESS;
}