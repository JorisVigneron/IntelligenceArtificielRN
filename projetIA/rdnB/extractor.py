import random
#
# Parser de fichier .data fournit par l'enseignant
#
def extractor_1(file):
    our_file = open(file,"r")
    liste = []
    while 1:
        data = our_file.readline()
        lis = []
        t = ''
        i = 0
        while i != len(data):
            if data[i] == 'N' or data[i] == 'B':
                t = t + '0'
                i = i + 1
            elif data[i] == 'R' or data[i] == 'M':
                t = t + '1'
                i = i + 1
            elif data[i] == ',':
                lis.append(t)
                t =''
                i += 1
            else:
                t = t + data[i]
                i += 1
        
        if not data:
            break
        liste.append(lis)
    our_file.close()
    return liste

#
# Extrait le batch des entrees sous forme de liste de string
#

def extractor_utils(liste):
    l = []
    a = []
    for i in range(0,len(liste)):
        a = liste[i][2::]
        w = [a]
        l.extend(w)
    return l

def upgrade_list(liste1, liste2, size):
    l1 = liste1
    l2 = liste2
    for i in range(len(liste1),size):
        a = random.randint(2,6)
        b = random.randint(1,10)
        k = []
        for j in range(len(liste1[1])):
            k.append(liste1[b][j]*(a/100))
        l1.append(k)
        l2.append(liste2[b])
    return l1, l2
#
# Extrait le batch des sorties sous forme de liste de string
#

def extractor_res(liste):
    l = []
    a = []
    for i in range(0,len(liste)):
        a = liste[i][1]
        l.extend(a)
    return l

def double_sortie(liste):
    l2 = []
    for i in range(0, len(liste)):
        l1 = []	
        if liste[i] == 1 :
            l1.append(1)
            l1.append(0)
        elif liste[i] == 0 :
                l1.append(0)
                l1.append(1)
        l2.append(l1)
        del l1
    return l2

#
# Convertion string to int pour les listes
#
    
def list_list_string_to_int(liste):
    for i in range(len(liste)):
        for j in range(len(liste[1])):
            liste[i][j] = float(liste[i][j])
    return liste

def list_string_to_int(liste):
    for i in range(len(liste)):
        liste[i] = int(liste[i])
    return liste


#print(a)
#print("\n")
#print(b)

