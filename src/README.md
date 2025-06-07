## 1. onde fazer?
da pra fazer tudo isso pelo terminal do proprio vscode, mas a ideia é fazer pelo cmd do pc 

## 2. Onde começa

em ambos os locais tem que verificar se o diretorio ta certo. 
é pra ser algo tipo ```C:\.....\trab_estatistica```

se não for ai vc vai abrindo as pastas pelo terminal 

 - ver as pastas: ```ls```
 - entrar nas pastas:```cd nome da pasta```

obs: se colocar o caminho todo no cd tbm funciona, tipo ```cd C:\Users\onedrive\facul\trab_estatistica```

## 3. Configurar Ambiente Virtual
Instalando o ambiente:

```python -m venv .venv```

Ativando o ambiente: 

```.venv\Scripts\activate```

Baixando as bibliotecas necessárias:
```pip install -r requirements.txt```

## 4. Executar o Script Principal

```python -m src.main```

## 5. Desativar o Ambiente Virtual
```deactivate```

## 6. Mostrar os Graficos

```python metrics/graficos.py```
