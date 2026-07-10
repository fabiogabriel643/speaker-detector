"""
nome = "Fabio"
idade = 24
altura = 1.78

print(f"Meu nome eh {nome}, e tenho {idade} anos")

a = 10
b = 3
print(a + b)

c = "10"
d = "3"

print(a + int(d))



if idade >= 18:
    print("Maior de idade")
else:
    print("menor de idade")

for i in range(2, 8):
    print(i)


for i in range(11):
    if 0 % 2 == 0:
        print("é par", i)
    
    
   
frutas = ["uva", "banana"]
idades = [23,25,32]
nomes = ["fabio", "ana","marcos"]

for i,fruta in enumerate (frutas):
    print(f"Na posicao {i}, temos a {fruta}")
for nome in nomes:
    print(f"Olá {nome}!")

for fruta, age in zip(frutas, idades):
    print(f"{fruta} tem {age} anos")
    
numeros = [1,2,3,4,5,7,8,9,10] 
num_par = []
num_impar = []
for num in numeros:
    if num % 2 == 0:
        num_par.append(num)
    else:
        num_impar.append(num)
print(len(numeros),num_par)
print(len(num_impar))
print(sum(num_par))
soma_par = sum(num_par)
soma_impar = sum(num_impar)
total = soma_par + soma_impar

print(total)
print(sum(numeros))

def saudar(nomes):
    print(f"Ola {nomes}")
    
saudar("Fabio")

def soma(a ,b):
    return a + b

resultado = soma(5,8)
print(resultado)

def maior(a, b):
    if(a > b):
        return a
    else:
        return b
result = maior(8, 9)
print(f"{result} é maior")

segmentos = [
        {"inicio": 0, "fim": 30,"speaker": "SPEAKER_00"},
        {"inicio": 31, "fim": 60,"speaker": "SPEAKER_01"},
        {"inicio": 61, "fim": 90,"speaker": "SPEAKER_02"},     
    ]
def resumo_speakers(segmentos):
    
    for seg in segmentos:
        minutodeinicio = seg["inicio"] // 60
        segundosdeinicio = seg["inicio"] % 60
        minutofinal = seg["fim"] // 60
        segundofinal = seg["fim"] % 60
        print(f"{minutodeinicio:02d}:{segundosdeinicio:02d} -> {minutofinal:02d}:{segundofinal:02d} : {seg["speaker"]}")
        
resumo_speakers(segmentos)


cont = 0

while cont < 5:
    print(cont)
    cont += 1

def coordenadas():
    return -23.5, -46.6

lat, lon = coordenadas()
print(lat)
print(lon)


pessoa = {
    "nome": "Fabio",
    "idade": 36,
    "cidade": "Curitiba",
    "comida": "Hamburguer",
}

print(pessoa["nome"], pessoa["idade"], pessoa["cidade"])

pessoa["email"] = "fabio@gmail.com"
pessoa["idade"] = 24
print(pessoa)

nomes = ["Fabio", "Ana"]
nomes.append("Carlos")
print(nomes)
"""
from collections import Counter
"""
associaoces = {
    
}
deteccoes = [
    {"face":1, "speaker": "SPEAKER_00"},
    {"face":1, "speaker": "SPEAKER_00"},
    {"face":2, "speaker": "SPEAKER_01"},
    {"face":1, "speaker": "SPEAKER_00"}
]

for i in deteccoes:
    if i["face"] not in associaoces:
        associaoces[i["face"]] = []
    associaoces[i["face"]].append(i["speaker"])
    
print(associaoces)
"""

votos = ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
cont = Counter(votos)
'''print(cont)
print(cont.most_common(1))
'''

result = cont.most_common(1)
print(result[0])
print(result[0][0])