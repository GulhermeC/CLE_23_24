/**
 * @file sort_recursive.c
 * @brief Demonstrates bitonic sort on a set of integers read from a binary file.
 */

#include <stdio.h>
#include <stdlib.h>

/**
 * Compares two integers for ascending order.
 * 
 * @param a Pointer to the first integer to compare.
 * @param b Pointer to the second integer to compare.
 * @return Negative if a < b, zero if a == b, positive if a > b.
 */
int compare(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

/**
 * Compares two integers for descending order.
 * 
 * @param a Pointer to the first integer to compare.
 * @param b Pointer to the second integer to compare.
 * @return Negative if b < a, zero if b == a, positive if b > a.
 */
int compare_desc(const void *a, const void *b) {
    return (*(int *)b - *(int *)a);
}

/**
 * Swaps two integers if the first is greater than the second.
 * 
 * @param a Pointer to the first integer.
 * @param b Pointer to the second integer.
 */
void CAPS(int *a, int *b) {
    if (*a > *b) {
        int temp = *a;
        *a = *b;
        *b = temp;
    }
}

/**
 * Merges two bitonic sequences.
 * 
 * @param val Pointer to the array of integers.
 * @param low Starting index for merging.
 * @param count Number of elements to merge.
 * @param dir Direction for the merge (1 for ascending, 0 for descending).
 */
void bitonicMerge(int *val, int low, int count, int dir)
{
    if (count > 1)
    {
        int k = count / 2;
        for (int i = low; i < low + k; i++)
        {
            if (dir == (val[i] > val[i+k]))
            {
                int temp = val[i];
                val[i] = val[i+k];
                val[i + k] = temp;
            }
        }
        bitonicMerge(val, low, k, dir);
        bitonicMerge(val, low+k, k, dir);
    }
}

/**
 * Performs the bitonic sort on an array of integers.
 * 
 * @param val Pointer to the array of integers.
 * @param low Starting index for sorting.
 * @param count Number of elements to sort.
 * @param dir Sorting direction (1 for ascending, 0 for descending).
 */
void bitonicSort(int *val, int low, int count, int dir)
{
    if (count > 1)
    {
        int k = count / 2;

        // Sort in ascending order since dir here is 1
        bitonicSort(val, low, k, 1);

        // Sort in descending order since dir here is 0
        bitonicSort(val, low + k, k, 0);

        // Merge the whole array
        bitonicMerge(val, low, count, dir);
    }
}

/**
 * Main program to demonstrate bitonic sort.
 * Reads integers from a binary file specified by the user, sorts them, and prints the result.
 * 
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return Returns EXIT_SUCCESS upon success, or EXIT_FAILURE upon failure.
 */
int main(int argc, char **argv)
{
    /* Read file */

    FILE *filePtr;
    char fileName[] = "dataSet2/datSeq32.bin"; // Specify the path to your binary file
    int numberOfIntegers, i;
    int *val;

    // Open the file in binary read mode
    filePtr = fopen(fileName, "rb");
    if (filePtr == NULL) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    // Read the first integer to find out the number of integers in the file
    if (fread(&numberOfIntegers, sizeof(int), 1, filePtr) != 1) {
        perror("Error reading number of integers");
        fclose(filePtr);
        return EXIT_FAILURE;
    }

    // Dynamically allocate memory for the integers (excluding the first one)
    val = (int *)malloc(numberOfIntegers * sizeof(int));
    if (val == NULL) {
        perror("Memory allocation failed");
        fclose(filePtr);
        return EXIT_FAILURE;
    }

    // Read the integers into the array
    for (i = 0; i < numberOfIntegers; i++) {
        if (fread(&val[i], sizeof(int), 1, filePtr) != 1) {
            perror("Error reading an integer from the file");
            free(val); // Free allocated memory before exiting
            fclose(filePtr);
            return EXIT_FAILURE;
        }
    }

    // Close the file
    fclose(filePtr);
    

    /* Standard merge sorting */
    
    // Calculate the midpoint of the array to divide it into two halves
    int midPoint = numberOfIntegers / 2;

    // Sort the first half
    qsort(val, midPoint, sizeof(int), compare);

    // Sort the second half
    qsort(val + midPoint, numberOfIntegers - midPoint, sizeof(int), compare);

    printf("\nBefore sorting: \n");
    // Print the integers from the array
    for (i = 0; i < numberOfIntegers; i++) {
        if(i == numberOfIntegers/2)
        {
            printf("\n");
        }
        printf("%d\n", val[i]);
    }
    
    printf("\nAfter Standard merge sorting:\n");
    
    int N = numberOfIntegers;
    for (int m = 0; m < N/2; m++)
        for (int n = 0; (m + n) < N/2; n++)
            CAPS(&val[m+n], &val[N/2+n]);

    for (i = 0; i < numberOfIntegers; i++) {
        if(i == numberOfIntegers/2)
        {
            printf("\n");
        }
        printf("%d\n", val[i]);
    }

    /* After Standard merge sorting */
    for (i = 0; i < N - 1; i++) {
        if (val[i] > val[i + 1]) {
            printf("Error in position %d between element %d and %d\n", i, val[i], val[i + 1]);
            break;
        }
    }
    if (i == (N - 1)) {
        printf("Everything is OK with Standard merge sorting!\n");
    }


    /* Bitonic sorting */

    // Sort the first half
    qsort(val, midPoint, sizeof(int), compare);

    // Sort the second half
    qsort(val + midPoint, numberOfIntegers - midPoint, sizeof(int), compare_desc);

    printf("\nBefore bitonic sorting: \n");
    for (i = 0; i < numberOfIntegers; i++) {
        printf("%d\n", val[i]);
    }

    // Apply bitonic sort
    bitonicSort(val, 0, numberOfIntegers, 1); // Here, '1' means sorting in ascending order

    printf("\nAfter bitonic sorting:\n");
    for (i = 0; i < numberOfIntegers; i++) {
        printf("%d\n", val[i]);
    }

    /* After Bitonic sorting */
    for (i = 0; i < N - 1; i++) {
        if (val[i] > val[i + 1]) {
            printf("Error in position %d between element %d and %d\n", i, val[i], val[i + 1]);
            break;
        }
    }
    if (i == (N - 1)) {
        printf("Everything is OK with Bitonic sorting!\n");
    }

    // Free the dynamically allocated memory
    free(val);
    return EXIT_SUCCESS;
}