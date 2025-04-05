import os

class Colors:
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BRIGHT = '\033[1m'

clear = lambda: os.system('cls' if os.name == 'nt' else 'clear')
clear()

M = [[1,7,3],[4,5,6],[7,8,9]]
I = [[1,0,0],[0,1,0],[0,0,1]]   

def pop(A, i, j):
    n = len(A)
    B = []
    for x in range(n):
        if(x != i):
            C = []
            for y in range(n):
                if(y != j):
                    C.append(A[x][y])
            B.append(C)
    return B
    
def det(M):
    if len(M) == 1:
        return M[0][0]
    if len(M) == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]   
    d = 0
    for i in range(len(M)):
        d += (-1)**(i) * M[i][0] * det(pop(M, i, 0))
    return d

def com(M):
    n = len(M)
    B = []
    for x in range(n):
        C = []
        for y in range(n):
            C.append((-1)**(x+y)*det(pop(M, x, y)))
        B.append(C)
        
    return B

def transpose(M):
    n = len(M)
    B = []
    for x in range(n):
        C = []
        for y in range(n):
            C.append(M[y][x])
        B.append(C)
    return B

def multiply_from_scal(A, K):
    n = len(A)
    B = []
    for i in range(n):
        C = []
        for j in range(n):
            C.append(K*A[i][j])
        B.append(C)
    return B

def multiply_from_matrice(A, B):
    n = len(A)
    C = []
    for i in range(n):
        D = []
        for j in range(n):
            D.append(0)
        C.append(D)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k]*B[k][j]
    return C

def inverse(M):
    if(det(M) == 0):
        return "X"
    return multiply_from_scal(transpose(com(M)), 1/det(M))

def print_matrice(M, color=Colors.WHITE):
    print(color + "┌" + "─" * (len(M) * 8 - 1) + "┐")
    for x in range(len(M)):
        print(color + "│ ", end="")
        for y in range(len(M[x])):
            if isinstance(M[x][y], float):
                print(f"{M[x][y]:6.2f}", end=" ")
            else:
                print(f"{M[x][y]:6}", end=" ")
        print(color + "│")
    print(color + "└" + "─" * (len(M) * 8 - 1) + "┘" + Colors.RESET)
    print()

print(Colors.CYAN + "Matrice M:" + Colors.RESET)
print_matrice(M, Colors.CYAN)

print(Colors.YELLOW + "Determinant de M: " + Colors.BRIGHT + str(det(M)) + Colors.RESET + "\n")

print(Colors.MAGENTA + "Coomatrice M:" + Colors.RESET)
print_matrice(com(M), Colors.MAGENTA)

print(Colors.GREEN + "Determinant de I: " + Colors.BRIGHT + str(det(I)) + Colors.RESET + "\n")

print(Colors.BLUE + "Coomatrice I:" + Colors.RESET)
print_matrice(com(I), Colors.BLUE)

print(Colors.RED + "Transpose de M:" + Colors.RESET)
print_matrice(transpose(M), Colors.RED)

print(Colors.GREEN + "Inverse de M:" + Colors.RESET)
print_matrice(inverse(M), Colors.GREEN)
