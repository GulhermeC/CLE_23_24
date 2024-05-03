#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#define MPI_CHECK(call) \
    do { \
        int err = call; \
        if (err != MPI_SUCCESS) { \
            fprintf(stderr, "MPI error %d in file '%s' at line %i.\n", err, __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, err); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void bitonic_merge(int *data, int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            if ((i + k) < (low + cnt)) { // Ensuring the index is within bounds
                if (dir == (data[i] > data[i + k])) {
                    int temp = data[i];
                    data[i] = data[i + k];
                    data[i + k] = temp;
                }
            }
        }
        bitonic_merge(data, low, k, dir);
        bitonic_merge(data, low + k, k, dir);
    }
}

void bitonic_sort(int *data, int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonic_sort(data, low, k, 1);        // Sort in ascending order
        bitonic_sort(data, low + k, k, 0);    // Sort in descending order
        bitonic_merge(data, low, cnt, dir);   // Merge whole sequence into one direction
    }
}

int main (int argc, char *argv[])
{
   MPI_Comm presentComm, nextComm;
   MPI_Group presentGroup, nextGroup;
   int gMemb[8];
   int rank, nProc, nProcNow, nIter;
   int *sendData = NULL, *recData = NULL;
   int i, j, length = 0;
   FILE *file;

   MPI_Init (&argc, &argv);
   MPI_Comm_rank (MPI_COMM_WORLD, &rank);
   MPI_Comm_size (MPI_COMM_WORLD, &nProc);
   MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);


   if (rank == 0 && (nProc & (nProc - 1)) != 0) {
        printf("Number of processes must be a power of two.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

   if (rank == 0) {
        if (argc < 2) {
            printf("Usage: %s <filename>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }

        file = fopen(argv[1], "rb");
        if (!file) {
            perror("File opening failed");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }

        // Read the length of data from the file
        fread(&length, sizeof(int), 1, file);

        sendData = malloc(length * sizeof(int));
        if (!sendData) {
            printf("Memory allocation failed\n");
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }

        // Read the data into sendData
        fread(sendData, sizeof(int), length, file);
        fclose(file);
        printf("Data read by process %d with total length %d.\n", rank, length);
    
    }

    // Broadcast the length to all processes
    MPI_CHECK(MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD));

    if (recData != NULL) {
        free(recData);
        recData = NULL;
    }
    
    recData = malloc((length / nProc) * sizeof(int));
    if (!recData) {
        perror("Memory allocation for recData failed");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    nIter = (int) (log2(nProc) + 1.1);

    if ((length & (length - 1)) != 0) {
        printf("Number of elements must be a power of two.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    nProcNow = nProc;
    presentComm = MPI_COMM_WORLD;
    MPI_CHECK(MPI_Comm_group(presentComm, &presentGroup));

   for (i = 0; i < 8; i++)
     gMemb[i] = i;

   for (j = 0; j < nIter; j++)
   { 
    if (j > 0)
        { 
            MPI_Group_incl (presentGroup, nProcNow, gMemb, &nextGroup);
          MPI_Comm_create(presentComm, nextGroup, &nextComm);
          MPI_Group_free(&presentGroup);
          if (presentComm != MPI_COMM_WORLD) {
            MPI_Comm_free(&presentComm); // Free the previous communicator
            }
          if (nextComm == MPI_COMM_NULL) {
            if (recData != NULL) {
                free(recData);
                recData = NULL;
            }
            continue;
        }
          presentGroup = nextGroup;
          presentComm = nextComm;
          if (rank >= nProcNow) {
            if (recData != NULL) {
                free(recData);
            }
            MPI_Finalize();
            return EXIT_SUCCESS;
            }
        }
    
    if (presentComm != MPI_COMM_NULL) {
        MPI_CHECK(MPI_Comm_size (presentComm, &nProc));
        MPI_CHECK(MPI_Scatter (sendData, length / nProcNow, MPI_INT, recData, length / nProcNow, MPI_INT, 0, presentComm));
        printf ("Scattered data received by process %d with length = %d for a group of %d process(es).\n", rank, (length / nProcNow), nProc);
    }
     bitonic_sort(recData, 0, length / nProc, 1);
     
    int partner;
    for (int k = 2; k <= nProc; k = k * 2) {
        for (int j = k >> 1; j > 0; j = j >> 1) {
            partner = rank ^ j;
            if (partner > rank) {
                if ((rank & k) == 0) {
                    // Send the lower half and keep the higher half if the data is smaller
                    MPI_CHECK(MPI_Sendrecv_replace(recData, length / nProc, MPI_INT, partner, 0, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    bitonic_merge(recData, 0, length / nProc, 1);
                } else {
                    // Send the higher half and keep the lower half if the data is larger
                    MPI_CHECK(MPI_Sendrecv_replace(recData, length / nProc, MPI_INT, partner, 0, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    bitonic_merge(recData, 0, length / nProc, 0);
                }
            } else {
                if ((rank & k) == 0) {
                    // Send the higher half and keep the lower half if the data is smaller
                    MPI_CHECK(MPI_Sendrecv_replace(recData, length / nProc, MPI_INT, partner, 0, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    bitonic_merge(recData, 0, length / nProc, 1);
                } else {
                    // Send the lower half and keep the higher half if the data is larger
                    MPI_CHECK(MPI_Sendrecv_replace(recData, length / nProc, MPI_INT, partner, 0, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    bitonic_merge(recData, 0, length / nProc, 0);
                }
            }
        }
    }

     MPI_CHECK(MPI_Gather (recData, length / nProcNow, MPI_INT, sendData, length / nProcNow, MPI_INT, 0, presentComm));
     if (rank == 0)
        printf ("Gathered data received by process 0 with length = %d for a group of %d process(es).\n", length, nProc);
     
     // Reduce the number of processes for the next iteration
     MPI_CHECK(MPI_Barrier(presentComm));
     nProcNow = nProcNow >> 1;
   }

    if (rank == 0) {
       int sorted = 1;  // Assume sorted unless proven otherwise
       for (i = 1; i < length; i++) {
           if (sendData[i - 1] > sendData[i]) {
               sorted = 0;
               break;
           }
       }
       if (sorted)
           printf("Sorting successful!\n");
       else
           printf("Sorting failed!\n");
   }
    if (presentComm != MPI_COMM_WORLD) {
        MPI_Comm_free(&presentComm); // Ensure all custom communicators are freed
    }
    if (presentComm != MPI_COMM_WORLD && presentComm != MPI_COMM_NULL) {
        MPI_Comm_free(&presentComm);
    }

   free(recData);
   if (rank == 0) {
       free(sendData);
   }
   
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();

   return EXIT_SUCCESS;
}