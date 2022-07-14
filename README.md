# Progetto di Human Data Science - Halo Ranking
Questo repository contiene il software utilizzato per svolgere il progetto per il corso di Human Data Science dell'università di Bologna (a.a. 2021/2022). 

Il repository è così strutturato:

* file `.cs`: codice sorgente
* la cartella `scripts` contiene alcuni script python impiegati
* il progetto si appoggia su un database mongodb (i cui parametri di connessione vanno specificati in `Program.cs`) con una collection composta da una lista di match; un esempio di match si può trovare nella cartella `example`. **Non** viene fornito un database a cui collegarsi.

## Uso
Questo software permette di calcolare le skill di giocatori, data una lista di partite del gioco di Halo Infinite e settati i parametri nella funzione `InferSkillsAndParameters()` del file `ParamsCalculator.cs`.

#### Requisiti
* dotnet > 6.0
* istanza inizializzata di un database mongodb

#### Avvio
Per usarlo, digitare 

```
dotnet restore
dotnet run
```