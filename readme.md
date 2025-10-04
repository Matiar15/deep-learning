# Projekt 3 – Autoencoder / VAE: Redukcja wymiarowości i generowanie danych

Projekt 3 dotyczy autoenkoderów i wariacyjnych autoenkoderów, których zadaniem jest redukcja wymiarowości oraz generowanie danych. Celem projektu jest zbudowanie klasycznego autoenkodera do kompresji i rekonstrukcji obrazów oraz wariacyjnego autoenkodera do generowania nowych przykładów. Dane wykorzystywane w projekcie pochodzą ze zbiorów MNIST i Fashion-MNIST, które zawierają odpowiednio obrazy cyfr oraz ubrań w rozdzielczości 28 na 28 pikseli. Oba zbiory są dostępne w bibliotekach Keras i TorchVision.

W ramach projektu należy zaimplementować klasyczny autoenkoder i ocenić jakość jego rekonstrukcji. Następnie należy przeprowadzić redukcję wymiarowości i wizualizację przestrzeni latentnej przy użyciu metod PCA lub t-SNE. Kolejnym etapem jest implementacja wariacyjnego autoenkodera i generacja nowych obrazów. Ostatecznym krokiem jest porównanie klasycznego autoenkodera i wariacyjnego autoenkodera pod względem jakości rekonstrukcji oraz zdolności generacyjnych.

W projekcie można wykorzystać różne zbiory danych. Podstawowymi są MNIST i Fashion-MNIST. Alternatywnie można użyć zbiorów CIFAR-10 lub CIFAR-100 zawierających kolorowe obrazy, zbioru CelebA zawierającego zdjęcia twarzy, a także zbiorów medycznych MedMNIST v2 lub HAM10000. Wybór zbioru zależy od celu eksperymentu, a różne zestawy danych pozwalają przetestować działanie modeli w zróżnicowanych warunkach.


# Raport 1 – Przygotowanie danych

## Cel
Pokazać, jak przygotowano zbiór danych do trenowania Autoenkodera (AE) i Wariacyjnego Autoenkodera (VAE).

## Zawartość raportu

- **Opis datasetu**  
  Zbiory danych: MNIST / Fashion-MNIST  
  Liczba klas: 10  
  Liczba obrazów: zależna od zbioru (np. 60 000 obrazów treningowych, 10 000 testowych)

- **Normalizacja danych**  
  Dane należy znormalizować do zakresu 0–1.

- **Schemat podziału**  
  Podział danych na zbiory:
    - Train
    - Validation
    - Test

- **Przykładowe obrazy z datasetu**  
  Kilka wierszy obrazów cyfr lub ubrań w celu wizualnej prezentacji danych.
