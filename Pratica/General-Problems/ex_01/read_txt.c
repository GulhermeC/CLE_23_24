#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ctype.h>


// Function: isWordBoundary
// Description: Checks if a single-byte character is considered a word boundary.
// This includes whitespace characters (space, tab, newline, carriage return),
// punctuation symbols (., ,, :, ;, ?, !), separation symbols (-, ", [, ], (, )),
// and the apostrophe ('), treating it as a word boundary for simplicity.
// Parameters:
// - c: The character to check.
// Returns: 1 if the character is a word boundary, 0 otherwise.
int isWordBoundary(char c)
{
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
        c == '-' || c == '"' || c == '[' || c == ']' || c == '(' || c == ')' ||
        c == '.' || c == ',' || c == ':' || c == ';' || c == '?' || c == '!' ||
        c == '\'' 
    ) {
        return 1;
    }
    return 0;
}

// Function: isUTF8WordBoundary
// Description: Checks for specific multi-byte UTF-8 sequences that represent word boundaries,
// specifically the dash (—, encoded as 0xE2 0x80 0x93) and the ellipsis (…, encoded as 0xE2 0x80 0xA6).
// These characters are considered as punctuation that separates words.
// Parameters:
// - ptr: Pointer to the current position in a string where a potential multi-byte
//        UTF-8 sequence starts.
// Returns: 1 if the sequence at the pointer represents a UTF-8 encoded word boundary, 0 otherwise.
int isUTF8WordBoundary(char *ptr) {
    // Check for dash (—)
    if ((unsigned char)ptr[0] == 0xE2 && (unsigned char)ptr[1] == 0x80 && (unsigned char)ptr[2] == 0x93) {
        return 1;
    }
    // Check for ellipsis (…)
    if ((unsigned char)ptr[0] == 0xE2 && (unsigned char)ptr[1] == 0x80 && (unsigned char)ptr[2] == 0xA6) {
        return 1;
    }
    return 0;
}

// Function:
int isConsonant(char c) {
    c = tolower(c);
    return c == 'b' || c == 'c' || c == 'd' || c == 'f' || c == 'g' ||
           c == 'h' || c == 'j' || c == 'k' || c == 'l' || c == 'm' ||
           c == 'n' || c == 'p' || c == 'q' || c == 'r' || c == 's' ||
           c == 't' || c == 'v' || c == 'w' || c == 'x' || c == 'y' || c == 'z';
}

int main(int argc, char **argv)
{
    char line[256] = {0};
    FILE *inputfp1;
    int word_count = 0;
    //int double_cons = 0;
    int total = 0;
    bool in_word = false;
    printf("\n");

    if (argc < 2)
    {
        printf("ERROR No text files to read\n");
        exit(1);

    }

    for (int i = 1; i < argc; ++i)
    {
        inputfp1 = fopen(argv[i], "r");
        
        if (inputfp1 == NULL)
        {
            printf("ERROR Opening file %s\n", argv[i]);
            exit(0);
        }

        word_count = 0;
        //double_cons = 0;

        while(fgets(line, sizeof(line), inputfp1) != NULL)
        {
            for (int j = 0; line[j]; ++j)
            {
                if (isWordBoundary(line[j]) || (line[j] && line[j+1] && line[j+2] && isUTF8WordBoundary(&line[j])))
                {
                    if (in_word)
                    {
                        in_word = false;
                        word_count++;

                        if (isUTF8WordBoundary(&line[j])) j += 2;  // Skip the rest of the multi-byte character
                    }
                }
                else if (!in_word)
                {
                    in_word = true;
                }
            }

        }
        fclose(inputfp1);

        if (in_word)
        {
            word_count++;
            in_word = false;
        }

        printf("Total word in file %s: %d\n", argv[i], word_count);
        total = total + word_count;
    }

    printf("Total words: %d\n", total);
    return 0;
}