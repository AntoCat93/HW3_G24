'''
We use a recursive function to find the max lenght of a substring that can be read in the same way forward and backwards:

- at the beginning the function receives the complete string, if the lenght of the string is 1, the result is just 1
- if the length of the string is 2 and the 2 letters are the same, the result is just 2
- if the first letter and the last letter are the same, we return 2 plus the function executed again with the string without the first and the last letter
- else return the max between the same function executed with the string without the last letter and the string without the first letter.
With the recursion the function will be executed untill the string has lenght equal to 1 or equal to 2.
'''
def findmaxsequence(string):
    i = 0
    j = len(string) - 1
    if string[i] == string[j] and i == j:
        return 1
    elif string[i] == string[j] and i+1 == j:
        return 2
    elif string[i] == string[j]:
        return  2 + findmaxsequence(string[i+1:j])
    else:
        return max(findmaxsequence(string[i:j]), findmaxsequence(string[i+1:j+1]))

s = input()
print(findmaxsequence(s))