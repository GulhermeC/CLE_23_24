#include <stdio.h>
#include <stdlib.h>

int compare(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

int compare_desc(const void *a, const void *b) {
    return (*(int *)b - *(int *)a);
}


void CAPS(int *a, int *b) {
    if (*a > *b) {
        int temp = *a;
        *a = *b;
        *b = temp;
    }
}


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

    /* Bitonic sorting */

    // Sort the first half
    qsort(val, midPoint, sizeof(int), compare);

    // Sort the second half
    qsort(val + midPoint, numberOfIntegers - midPoint, sizeof(int), compare_desc);

    printf("\nBefore sorting: \n");
    // Print the integers from the array
    for (i = 0; i < numberOfIntegers; i++) {
        if(i == numberOfIntegers/2)
        {
            printf("\n");
        }
        printf("%d\n", val[i]);
    }


    // Free the dynamically allocated memory
    free(val);
}