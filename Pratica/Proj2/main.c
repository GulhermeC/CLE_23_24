#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <time.h>

#define MPI_CHECK(call)                                                                          \
    do                                                                                           \
    {                                                                                            \
        int err = call;                                                                          \
        if (err != MPI_SUCCESS)                                                                  \
        {                                                                                        \
            fprintf(stderr, "MPI error %d in file '%s' at line %i.\n", err, __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, err);                                                      \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    } while (0)

/**
 * @brief Recursively merges subsequences in bitonic order using the direction parameter.
 * @param data Pointer to the array containing the data to merge.
 * @param low Starting index of the subsequence within the array.
 * @param cnt Number of elements in the subsequence to merge.
 * @param dir Direction to merge the data (1 for ascending, 0 for descending).
 */
void bitonic_merge(int *data, int low, int cnt, int dir)
{
    if (cnt > 1)
    {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++)
        {
            if ((i + k) < (low + cnt))
            { // Ensuring the index is within bounds
                if (dir == (data[i] > data[i + k]))
                {
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


/**
 * @brief Sorts the data recursively in a bitonic sequence and then merges.
 * @param data Pointer to the array containing the data to sort.
 * @param low Starting index of the subsequence within the array.
 * @param cnt Number of elements in the subsequence to sort.
 * @param dir Direction to sort the data (1 for ascending, 0 for descending).
 */
void bitonic_sort(int *data, int low, int cnt, int dir)
{
    if (cnt > 1)
    {
        int k = cnt / 2;
        bitonic_sort(data, low, k, 1);      // Sort in ascending order
        bitonic_sort(data, low + k, k, 0);  // Sort in descending order
        bitonic_merge(data, low, cnt, dir); // Merge whole sequence into one direction
    }
}

/**
 *  @brief Get the process time that has elapsed since last call of this time.
 *
 *  @return process elapsed time
 */
double get_delta_time(void)
{
    static struct timespec t0, t1;

    t0 = t1;
    if(clock_gettime (CLOCK_MONOTONIC, &t1) != 0) {
        perror ("clock_gettime");
        exit(3);
    }
    return (double) (t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double) (t1.tv_nsec - t0.tv_nsec);
}


int main(int argc, char *argv[])
{
    MPI_Comm presentComm, nextComm;
    MPI_Group presentGroup, nextGroup;
    int gMemb[8];
    int rank, nProc, nProcNow, length, nIter;
    int *sendData = NULL, *recData;
    int i, j;
    FILE *file;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);

    if(rank == 0) {
        (void) get_delta_time();
        file = fopen(argv[1], "rb");
        if (!file)
        {
            perror("File opening failed");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }
        fread(&length, sizeof(int), 1, file);
    }

    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if ((length & (length - 1)) != 0)
    {
        printf("Number of elements must be a power of two.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    nIter = (int)(log2(nProc) + 1.1);
    recData = malloc(length * sizeof(int));
    if (rank == 0)
    {
        sendData = malloc(length * sizeof(int));
        fread(sendData, sizeof(int), length, file);
        fclose(file);
    }

    nProcNow = nProc;
    presentComm = MPI_COMM_WORLD;
    MPI_Comm_group(presentComm, &presentGroup);
    for (i = 0; i < 8; i++)
        gMemb[i] = i;

    for (j = 0; j < nIter; j++)
    {
        if (j > 0)
        {
            MPI_Group_incl(presentGroup, nProcNow, gMemb, &nextGroup);
            MPI_Comm_create(presentComm, nextGroup, &nextComm);
            presentGroup = nextGroup;
            presentComm = nextComm;
            if (rank >= nProcNow)
            {
                if (recData != NULL)
                {
                    free(recData);
                }
                MPI_Finalize();
                return EXIT_SUCCESS;
            }
        }

        if (presentComm != MPI_COMM_NULL)
        {
            MPI_CHECK(MPI_Comm_size(presentComm, &nProc));
            MPI_CHECK(MPI_Scatter(sendData, length / nProcNow, MPI_INT, recData, length / nProcNow, MPI_INT, 0, presentComm));
            printf("Scattered data received by process %d with length = %d for a group of %d process(es).\n", rank, (length / nProcNow), nProc);
        }

        bitonic_sort(recData, 0, length / nProc, 1);

        for (int k = 2; k <= nProc; k = k * 2)
        {
            for (int j = k >> 1; j > 0; j = j >> 1)
            {
                
                if ((rank & k) == 0)
                {
                    bitonic_merge(recData, 0, length / nProc, 1);
                }
                else
                {
                    bitonic_merge(recData, 0, length / nProc, 0);
                }
            }
        }

        MPI_CHECK(MPI_Gather(recData, length / nProcNow, MPI_INT, sendData, length / nProcNow, MPI_INT, 0, presentComm));
        if (rank == 0)
            printf("Gathered data received by process 0 with length = %d for a group of %d process(es).\n", length, nProc);

        // Reduce the number of processes for the next iteration
        MPI_CHECK(MPI_Barrier(presentComm));
        nProcNow = nProcNow >> 1;
    }

    if (rank == 0)
    {
        int sorted = 1;
        for (i = 1; i < length; i++)
        {
            if (sendData[i - 1] > sendData[i])
            {
                sorted = 0;
                break;
            }
        }
        if (sorted)
            printf("The sequence is properly sorted.\n");
        else
            printf("The sequence is NOT properly sorted.\n");
    }

    free(recData);
    if (rank == 0)
    {
        free(sendData);
        printf("time elapsed: %.6fs\n",get_delta_time());
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}